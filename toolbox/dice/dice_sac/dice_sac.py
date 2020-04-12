import ray
from ray.rllib.agents.dqn.dqn import update_target_if_needed
from ray.rllib.agents.sac.sac import SACTrainer

import toolbox.dice.utils as constants
from toolbox.dice.dice import validate_config, setup_policies_pool, \
    make_policy_optimizer_tnbes
from toolbox.dice.dice_sac.dice_sac_config import dice_sac_default_config
from toolbox.dice.dice_sac.dice_sac_policy import DiCESACPolicy


def after_optimizer_step(trainer, fetches):
    # Original SAC operation
    update_target_if_needed(trainer, fetches)

    # only update the policies pool if used DELAY_UPDATE, otherwise
    # the policies_pool in each policy is simply not used, so we don't
    # need to update it.
    if trainer.config[constants.DELAY_UPDATE]:
        if trainer.workers.remote_workers():
            weights = ray.put(trainer.workers.local_worker().get_weights())
            for e in trainer.workers.remote_workers():
                e.set_weights.remote(weights)

            def _delay_update_for_worker(worker, worker_index):
                worker.foreach_policy(lambda p, _: p.update_target_network())

            trainer.workers.foreach_worker_with_index(_delay_update_for_worker)


# TODO the policy is not finish yet.

DiCETrainer = SACTrainer.with_updates(
    name="DiCETrainer",
    default_config=dice_sac_default_config,
    default_policy=DiCESACPolicy,
    get_policy_class=lambda _: DiCESACPolicy,

    # FIXME finished but not tested
    after_init=setup_policies_pool,
    after_optimizer_step=after_optimizer_step,

    # FIXME not finish
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer_tnbes,
)
