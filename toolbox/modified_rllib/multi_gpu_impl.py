"""Copied from rllib. Our main modification is that we allow
no-gpu-split data input. So overcome troubles if we want to input
strange shape data which can not be spread across gpus."""
import logging

from ray.rllib.optimizers.multi_gpu_impl import average_gradients, \
    make_divisible_by, Tower
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.debug import log_once, summarize

tf = try_import_tf()

# Variable scope in which created variables will be placed under
TOWER_SCOPE_NAME = "tower"

logger = logging.getLogger(__name__)


class LocalSyncParallelOptimizerModified(object):
    """Optimizer that runs in parallel across multiple local devices.

    LocalSyncParallelOptimizer automatically splits up and loads training data
    onto specified local devices (e.g. GPUs) with `load_data()`. During a call
    to `optimize()`, the devices compute gradients over slices of the data in
    parallel. The gradients are then averaged and applied to the shared
    weights.

    The data loaded is pinned in device memory until the next call to
    `load_data`, so you can make multiple passes (possibly in randomized order)
    over the same data once loaded.

    This is similar to tf.train.SyncReplicasOptimizer, but works within a
    single TensorFlow graph, i.e. implements in-graph replicated training:

      https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer

    Args:
        optimizer: Delegate TensorFlow optimizer object.
        devices: List of the names of TensorFlow devices to parallelize over.
        input_placeholders: List of input_placeholders for the loss function.
            Tensors of these shapes will be passed to build_graph() in order
            to define the per-device loss ops.
        rnn_inputs: Extra input placeholders for RNN inputs. These will have
            shape [BATCH_SIZE // MAX_SEQ_LEN, ...].
        max_per_device_batch_size: Number of tuples to optimize over at a time
            per device. In each call to `optimize()`,
            `len(devices) * per_device_batch_size` tuples of data will be
            processed. If this is larger than the total data size, it will be
            clipped.
        build_graph: Function that takes the specified inputs and returns a
            TF Policy instance.
    """

    def __init__(
            self,
            optimizer,
            devices,
            input_placeholders_split,
            input_placeholders_nosplit,
            input_names,
            rnn_inputs,
            max_per_device_batch_size,
            build_graph,
            grad_norm_clipping=None
    ):
        self.optimizer = optimizer
        self.devices = devices
        self.max_per_device_batch_size = max_per_device_batch_size

        assert not rnn_inputs  # just a workaround, do not consider RNN.

        self.input_names = input_names
        self.loss_inputs = input_placeholders_split
        self.loss_inputs_nosplit = input_placeholders_nosplit

        self.all_loss_inputs = {}
        for pair_key in input_names:
            if pair_key in self.loss_inputs_nosplit:
                self.all_loss_inputs[pair_key
                                     ] = self.loss_inputs_nosplit[pair_key]
            else:
                self.all_loss_inputs[pair_key] = self.loss_inputs[pair_key]

        assert len(self.all_loss_inputs) == len(input_names)
        assert len(self.all_loss_inputs
                   ) == len(self.loss_inputs) + len(self.loss_inputs_nosplit)

        self.build_graph = build_graph

        # First initialize the shared loss network
        with tf.name_scope(TOWER_SCOPE_NAME):
            tmp_list = [
                self.all_loss_inputs[pair_key] for pair_key in self.input_names
            ]

            self._shared_loss = build_graph(tmp_list)
        shared_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name
        )

        # Then setup the per-device loss graphs that use the shared weights
        self._batch_index = tf.placeholder(tf.int32, name="batch_index")

        # Dynamic batch size, which may be shrunk if there isn't enough data
        self._per_device_batch_size = tf.placeholder(
            tf.int32, name="per_device_batch_size"
        )
        self._loaded_per_device_batch_size = max_per_device_batch_size

        # When loading RNN input, we dynamically determine the max seq len
        self._max_seq_len = tf.placeholder(tf.int32, name="max_seq_len")
        self._loaded_max_seq_len = 1

        # Split on the CPU in case the data doesn't fit in GPU memory.
        with tf.device("/cpu:0"):
            data_splits = zip(
                *[tf.split(ph, len(devices)) for ph in \
                  self.loss_inputs.values()])

        self._towers = []
        for device, device_placeholders in zip(self.devices, data_splits):
            self._towers.append(
                self._setup_device(
                    device, device_placeholders, len(input_placeholders_split)
                )
            )

        avg = average_gradients([t.grads for t in self._towers])
        if grad_norm_clipping:
            clipped = []
            for grad, _ in avg:
                clipped.append(grad)
            clipped, _ = tf.clip_by_global_norm(clipped, grad_norm_clipping)
            for i, (grad, var) in enumerate(avg):
                avg[i] = (clipped[i], var)

        # gather update ops for any batch norm layers. TODO(ekl) here we will
        # use all the ops found which won't work for DQN / DDPG, but those
        # aren't supported with multi-gpu right now anyways.
        self._update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name
        )
        for op in shared_ops:
            self._update_ops.remove(op)  # only care about tower update ops
        if self._update_ops:
            logger.debug(
                "Update ops to run on apply gradient: {}".format(
                    self._update_ops
                )
            )

        with tf.control_dependencies(self._update_ops):
            self._train_op = self.optimizer.apply_gradients(avg)

    def load_data(self, sess, inputs, state_inputs):
        """Bulk loads the specified inputs into device memory.

        The shape of the inputs must conform to the shapes of the input
        placeholders this optimizer was constructed with.

        The data is split equally across all the devices. If the data is not
        evenly divisible by the batch size, excess data will be discarded.

        Args:
            sess: TensorFlow session.
            inputs: List of arrays matching the input placeholders, of shape
                [BATCH_SIZE, ...].
            state_inputs: List of RNN input arrays. These arrays have size
                [BATCH_SIZE / MAX_SEQ_LEN, ...].

        Returns:
            The number of tuples loaded per device.
        """

        if log_once("load_data"):
            logger.info(
                "Training on concatenated sample batches:\n\n{}\n".format(
                    summarize(
                        {
                            "placeholders": self.loss_inputs,
                            "inputs": inputs,
                            "state_inputs": state_inputs
                        }
                    )
                )
            )

        feed_dict = {}
        assert len(self.all_loss_inputs) == len(inputs + state_inputs), \
            (self.all_loss_inputs, inputs, state_inputs)

        # Let's suppose we have the following input data, and 2 devices:
        # 1 2 3 4 5 6 7                              <- state inputs shape
        # A A A B B B C C C D D D E E E F F F G G G  <- inputs shape
        # The data is truncated and split across devices as follows:
        # |---| seq len = 3
        # |---------------------------------| seq batch size = 6 seqs
        # |----------------| per device batch size = 9 tuples

        if len(state_inputs) > 0:
            smallest_array = state_inputs[0]
            seq_len = len(inputs[0]) // len(state_inputs[0])
            self._loaded_max_seq_len = seq_len
        else:
            smallest_array = inputs[0]
            self._loaded_max_seq_len = 1

        sequences_per_minibatch = (
            self.max_per_device_batch_size // self._loaded_max_seq_len *
            len(self.devices)
        )
        if sequences_per_minibatch < 1:
            logger.warn(
                (
                    "Target minibatch size is {}, however the rollout sequence "
                    "length is {}, hence the minibatch size will be raised to "
                    "{}."
                ).format(
                    self.max_per_device_batch_size, self._loaded_max_seq_len,
                    self._loaded_max_seq_len * len(self.devices)
                )
            )
            sequences_per_minibatch = 1

        if len(smallest_array) < sequences_per_minibatch:
            # Dynamically shrink the batch size if insufficient data
            sequences_per_minibatch = make_divisible_by(
                len(smallest_array), len(self.devices)
            )

        if log_once("data_slicing"):
            logger.info(
                (
                    "Divided {} rollout sequences, each of length {}, among "
                    "{} devices."
                ).format(
                    len(smallest_array), self._loaded_max_seq_len,
                    len(self.devices)
                )
            )

        if sequences_per_minibatch < len(self.devices):
            raise ValueError(
                "Must load at least 1 tuple sequence per device. Try "
                "increasing `sgd_minibatch_size` or reducing `max_seq_len` "
                "to ensure that at least one sequence fits per device."
            )
        self._loaded_per_device_batch_size = (
            sequences_per_minibatch // len(self.devices) *
            self._loaded_max_seq_len
        )

        if len(state_inputs) > 0:
            # First truncate the RNN state arrays to the sequences_per_minib.
            state_inputs = [
                make_divisible_by(arr, sequences_per_minibatch)
                for arr in state_inputs
            ]
            # Then truncate the data inputs to match
            inputs = [arr[:len(state_inputs[0]) * seq_len] for arr in inputs]
            assert len(state_inputs[0]) * seq_len == len(inputs[0]), \
                (len(state_inputs[0]), sequences_per_minibatch, seq_len,
                 len(inputs[0]))
            for ph, arr in zip(self.all_loss_inputs.values(),
                               inputs + state_inputs):
                feed_dict[ph] = arr
            truncated_len = len(inputs[0])
        else:
            for (pair_key, ph), arr in zip(self.all_loss_inputs.items(),
                                           inputs + state_inputs):
                if pair_key in self.loss_inputs:
                    # If this data should be truncated, then truncate it.
                    truncated_arr = make_divisible_by(
                        arr, sequences_per_minibatch
                    )
                    feed_dict[ph] = truncated_arr
                    truncated_len = len(truncated_arr)
                else:
                    # otherwise just use the pure data.
                    feed_dict[ph] = arr

        sess.run([t.init_op for t in self._towers], feed_dict=feed_dict)

        self.num_tuples_loaded = truncated_len
        tuples_per_device = truncated_len // len(self.devices)
        assert tuples_per_device > 0, "No data loaded?"
        assert tuples_per_device % self._loaded_per_device_batch_size == 0
        return tuples_per_device

    def optimize(self, sess, batch_index):
        """Run a single step of SGD.

        Runs a SGD step over a slice of the preloaded batch with size given by
        self._loaded_per_device_batch_size and offset given by the batch_index
        argument.

        Updates shared model weights based on the averaged per-device
        gradients.

        Args:
            sess: TensorFlow session.
            batch_index: Offset into the preloaded data. This value must be
                between `0` and `tuples_per_device`. The amount of data to
                process is at most `max_per_device_batch_size`.

        Returns:
            The outputs of extra_ops evaluated over the batch.
        """
        feed_dict = {
            self._batch_index: batch_index,
            self._per_device_batch_size: self._loaded_per_device_batch_size,
            self._max_seq_len: self._loaded_max_seq_len,
        }
        for tower in self._towers:
            feed_dict.update(tower.loss_graph.extra_compute_grad_feed_dict())

        fetches = {"train": self._train_op}
        for tower in self._towers:
            fetches.update(tower.loss_graph._get_grad_and_stats_fetches())
        return sess.run(fetches, feed_dict=feed_dict)

    def get_common_loss(self):
        return self._shared_loss

    def get_device_losses(self):
        return [t.loss_graph for t in self._towers]

    def _setup_device(self, device, device_input_placeholders, num_data_in):
        assert num_data_in <= len(device_input_placeholders)
        with tf.device(device):
            with tf.name_scope(TOWER_SCOPE_NAME):
                device_input_batches = {}
                device_input_slices = {}
                for i, (pair_key,
                        ph) in enumerate(zip(self.loss_inputs.keys(),
                                             device_input_placeholders)):
                    current_batch = tf.Variable(
                        ph,
                        trainable=False,
                        validate_shape=False,
                        collections=[]
                    )
                    device_input_batches[pair_key] = current_batch
                    if i < num_data_in:
                        scale = self._max_seq_len
                        granularity = self._max_seq_len
                    else:
                        scale = self._max_seq_len
                        granularity = 1
                    current_slice = tf.slice(
                        current_batch, (
                            [self._batch_index // scale * granularity] +
                            [0] * len(ph.shape[1:])
                        ), (
                            [
                                self._per_device_batch_size // scale *
                                granularity
                            ] + [-1] * len(ph.shape[1:])
                        )
                    )
                    current_slice.set_shape(ph.shape)
                    device_input_slices[pair_key] = current_slice

                tmp_batch = []
                tmp_batch_map = {}
                for pair_key in self.input_names:
                    if pair_key in device_input_batches:
                        tmp_batch.append(device_input_batches[pair_key])
                        tmp_batch_map[pair_key] = device_input_batches[pair_key
                                                                       ]
                    else:
                        vb = tf.Variable(
                            self.loss_inputs_nosplit[pair_key],
                            trainable=False,
                            validate_shape=False,
                            collections=[]
                        )
                        vb.set_shape(self.loss_inputs_nosplit[pair_key].shape)
                        tmp_batch.append(vb)
                        tmp_batch_map[pair_key] = vb
                assert len(tmp_batch) == len(self.all_loss_inputs)

                tmp_slice = []
                for pair_key in self.input_names:
                    if pair_key in device_input_slices:
                        tmp_slice.append(device_input_slices[pair_key])
                    else:
                        tmp_slice.append(tmp_batch_map[pair_key])
                assert len(tmp_slice) == len(self.all_loss_inputs)

                graph_obj = self.build_graph(tmp_slice)
                device_grads = graph_obj.gradients(
                    self.optimizer, graph_obj._loss
                )
            return Tower(
                tf.group(*[batch.initializer for batch in tmp_batch]),
                device_grads, graph_obj
            )
