from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import collections
import distutils
import logging
import os
import pickle
import subprocess
import tempfile
import time
import uuid
from math import floor

import cv2
import numpy as np
import ray
from gym import logger, error
from ray.rllib.agents.registry import get_agent_class
from ray.tune.util import merge_dicts

ORIGINAL_VIDEO_WIDTH = 1920
ORIGINAL_VIDEO_HEIGHT = 1080

VIDEO_WIDTH_EDGE = 100
VIDEO_HEIGHT_EDGE = 60

VIDEO_WIDTH = ORIGINAL_VIDEO_WIDTH - 2 * VIDEO_WIDTH_EDGE
VIDEO_HEIGHT = ORIGINAL_VIDEO_HEIGHT - 2 * VIDEO_HEIGHT_EDGE


def build_config(ckpt, args_config):
    config = {"log_level": "ERROR"}
    if ckpt is not None:
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        # config = {"log_level": "ERROR"}
        # Load configuration from file
        config_dir = os.path.dirname(ckpt)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config.update(pickle.load(f))
    if "num_workers" in config:
        config["num_workers"] = min(1, config["num_workers"])
    config = merge_dicts(config, args_config or {})
    return config


def touch(path):
    open(path, 'a').close()


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


class VideoRecorder(object):
    """VideoRecorder renders a nice movie of a rollout, frame by frame. It
    comes with an `enabled` option so you can still use the same code
    on episodes where you don't want to record video.

    Note:
        You are responsible for calling `close` on a created
        VideoRecorder, or else you may leak an encoder process.

    Args:
        env (Env): Environment to take video of.
        path (Optional[str]): Path to the video file; will be randomly
        chosen if omitted.
        base_path (Optional[str]): Alternatively, path to the video file
        without extension, which will be added.
        metadata (Optional[dict]): Contents to save to the metadata file.
        enabled (bool): Whether to actually record video, or just no-op (for
        convenience)
    """

    def __init__(
            # self, env, path=None, metadata=None, base_path=None
            self,
            base_path,
            grids,
            FPS=50
    ):
        # self.grids = int | dict
        # if int, then it represent the number of videos
        # if dict, then it should be like {agent_name: (row, col)}
        self.frame_shape = None
        assert isinstance(grids, int) or isinstance(grids, dict)
        self.grids = grids
        self.frame_range = None
        self.scale = None
        self.last_frame = None
        self.num_cols = None
        self.num_rows = None

        required_ext = '.mp4'
        if base_path is not None:
            # Base path given, append ext
            path = base_path + required_ext
        else:
            # Otherwise, just generate a unique filename
            with tempfile.NamedTemporaryFile(suffix=required_ext,
                                             delete=False) as f:
                path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            hint = " HINT: The environment is text-only, therefore we're " \
                   "recording its text output in a structured JSON format." \
                if self.ansi_mode else ''
            raise error.Error(
                "Invalid path given: {} -- must have file "
                "extension {}.{}".format(self.path, required_ext, hint)
            )
        # Touch the file in any case, so we know it's present. (This
        # corrects for platform platform differences. Using ffmpeg on
        # OS X, the file is precreated, but not on Linux.
        touch(path)

        self.extra_num_frames = None
        self.frames_per_sec = FPS
        self.encoder = None  # lazily start the process

        logger.info('Starting new video recorder writing to %s', self.path)
        self.initialized = False

    def _put_text(
            self,
            timestep,
            text,
            pos,
            thickness=1,
            color=(0, 0, 0),
            rotate=False,
            not_scale=False
    ):
        # print("put text {} at pos {} at time {}".format(text, pos, timestep))
        if not text:
            return
        if timestep is None:
            timestep = list(range(len(self.background)))
        elif isinstance(timestep, int):
            timestep = [timestep]
        assert isinstance(timestep, list)

        if not rotate:
            for t in timestep:
                cv2.putText(
                    self.background[t], text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1 if not_scale else 0.38 * self.scale + 0.1, color,
                    thickness
                )
        else:
            text_img = np.zeros(
                (self.background[0].shape[1], self.background[0].shape[0], 4),
                dtype='uint8'
            )
            new_pos = ORIGINAL_VIDEO_HEIGHT - pos[1], pos[0]
            cv2.putText(
                text_img, text, new_pos, cv2.FONT_HERSHEY_SIMPLEX,
                1 if not_scale else 0.5 * self.scale + 0.15, color, thickness
            )
            # Only work for black background which is all zeros
            for t in timestep:
                self.background[t] += text_img[:, ::-1, :].swapaxes(0, 1)

    def _build_background(self, frames_dict):
        assert self.frames_per_sec is not None
        self.extra_num_frames = 5 * int(self.frames_per_sec)
        video_length = max(
            [
                len(frames_info['frames'])
                for frames_info in frames_dict.values()
            ]
        ) + self.extra_num_frames
        self.background = np.zeros(
            (video_length, ORIGINAL_VIDEO_HEIGHT, ORIGINAL_VIDEO_WIDTH, 4),
            dtype='uint8'
        )

    def _add_things_on_backgaround(self, frames_dict, extra_info_dict):
        # We can add all row / col name here!!!!
        drew_col = set()
        drew_row = set()
        for idx, (title, frames_info) in \
                enumerate(frames_dict.items()):
            column_str = frames_info['column']
            row_str = frames_info['row']
            rang = self.frame_range[idx] if frames_info['loc'] is None \
                else self._get_draw_range(*frames_info['loc'])
            if column_str is not None and column_str not in drew_col:
                drew_col.add(column_str)
                # DRAW COLUMN
                pos = rang['width'][0], int(VIDEO_HEIGHT_EDGE * 0.8)
                self._put_text(None, column_str, pos, color=(255, 255, 255))

            if row_str is not None and row_str not in drew_row:
                drew_row.add(row_str)
                # DRAW ROW
                pos = int(VIDEO_WIDTH_EDGE * 0.6), rang['height'][1]
                self._put_text(
                    None, row_str, pos, color=(255, 255, 255), rotate=True
                )

    def _get_location(self, index):
        assert isinstance(index, int)
        assert self.num_cols * self.num_rows >= index

        row_id = int(index / self.num_cols)
        col_id = int(index % self.num_cols)

        return row_id, col_id

    def _get_index(self, row_id, col_id):
        ret = row_id * self.num_cols + col_id
        assert ret < self.num_cols * self.num_rows
        return ret

    def _get_draw_range(self, row_id, col_id):
        idx = self._get_index(row_id, col_id)
        assert self.initialized
        return self.frame_range[idx]

    def _build_grid_of_frames(self, frames_dict, extra_info_dict):
        self._add_things_on_backgaround(frames_dict, extra_info_dict)
        for idx, (title, frames_info) in \
                enumerate(frames_dict.items()):

            frames = frames_info['frames']

            specify_loc = frames_info['loc'] is not None

            row_id, col_id = frames_info['loc'] if specify_loc \
                else self._get_location(idx)

            if row_id >= self.num_rows or col_id >= self.num_cols:
                logging.warning(
                    "The row {} and col {} is out of the bound of "
                    "[row={}, col={}]!!".format(
                        row_id, col_id, self.num_rows, self.num_cols
                    )
                )
                continue

            assert row_id * self.num_cols + col_id < len(self.frame_range), \
                "{}, {} (row={}, col={})".format(
                    row_id * self.num_cols + col_id, len(self.frame_range),
                    self.num_rows, self.num_cols)

            rang = self.frame_range[idx] if not specify_loc \
                else self.frame_range[row_id * self.num_cols + col_id]

            # TODO we can add async execution here
            height = rang["height"]
            width = rang["width"]

            def get_pos(left_ratio, bottom_ratio):
                assert 0 <= left_ratio <= 1
                assert 0 <= bottom_ratio <= 1
                return (
                    int(left_ratio * width[0] + (1 - left_ratio) * width[1]),
                    int(
                        bottom_ratio * height[0] +
                        (1 - bottom_ratio) * height[1]
                    )
                )

            if self.scale < 1:
                frames = [
                    cv2.resize(
                        frame, (
                            int(self.width * self.scale
                                ), int(self.height * self.scale)
                        ),
                        interpolation=cv2.INTER_CUBIC
                    ) for frame in frames
                ]
                frames = np.stack(frames)

            self.background[:len(frames), height[0]:height[1], width[0]:
                            width[1], 2::-1] = frames

            # filled the extra number of frames
            self.background[len(frames):, height[0]:height[1], width[0]:
                            width[1], 2::-1] = frames[-1]

            for information in extra_info_dict.values():
                if 'pos_ratio' not in information:
                    continue
                pos = get_pos(*information['pos_ratio'])
                value = information[title]
                if isinstance(value, list):
                    # filter out the empty list if any.
                    if not value:
                        continue
                    value_sequence = value
                    for timestep, value in enumerate(value_sequence):
                        text = information['text_function'](value)
                        self._put_text(timestep, text, pos)
                    text = information['text_function'](value_sequence[-1])
                    for timestep in range(len(value_sequence),
                                          len(self.background)):
                        self._put_text(timestep, text, pos)
                else:
                    text = information['text_function'](value)
                    self._put_text(None, text, pos)

        # self._add_things_on_backgaround(frames_dict)
        return self.background

    def generate_video(self, frames_dict, extra_info_dict):
        """Render the given `env` and add the resulting frame to the video."""
        logger.debug('Capturing video frame: path=%s', self.path)

        # assert isinstance(frames_dict, OrderedDict)
        # first_row = next(iter(frames_dict.values()))
        # assert isinstance(first_row, OrderedDict)

        # frames_dict = {VIDEO_NAME: {
        #       'frames': FRAME,
        #       'pos': (ROW, COL)
        #   },
        # ...,
        #       "row_names": [ROW1, ROW2, ..],
        #       "col_names": [COL1, COL2, ..],
        #       "frame_info": {'width':.., "height":.., }
        # }

        if not self.initialized:
            info = extra_info_dict['frame_info']
            # tmp_frame = list(frames_dict.values())[0][0]
            self.width = info['width']
            self.height = info['height']
            self._build_frame_range()
            self.initialized = True

        self._build_background(frames_dict)

        self._build_grid_of_frames(frames_dict, extra_info_dict)

        now = time.time()
        start = now
        for idx, frame in enumerate(self.background):
            if idx % 100 == 99:
                print(
                    "Current Frames: {}/{} (T +{:.1f}s Total {:.1f}s)".format(
                        idx + 1, len(self.background),
                        time.time() - now,
                        time.time() - start
                    )
                )
                now = time.time()
            self.last_frame = frame
            self._encode_image_frame(frame)

        self._close()
        return self.path

    def _close(self):
        """Make sure to manually close, or else you'll leak the encoder
        process"""

        # Add extra 5 seconds static frames to help visualization.
        # for _ in range(5 * int(self.frames_per_sec)):
        #     self._encode_image_frame(self.last_frame)

        if self.encoder:
            print('Closing video encoder: path={}'.format(self.path))
            self.encoder.close()
            self.encoder = None
        else:
            # No frames captured. Set metadata, and remove the empty output
            # file.
            os.remove(self.path)

    def _build_frame_range(self):
        # def center_range(center, rang):
        #     return [int(center - rang / 2), int(center + rang / 2)]

        specify_grids = not isinstance(self.grids, int)

        # if not specify_grids:
        wv_over_wf = VIDEO_WIDTH / self.width
        hv_over_hf = VIDEO_HEIGHT / self.height
        for potential in np.arange(1, 0, -0.01):
            # potential = 1, 0.9, ...
            if specify_grids:
                if wv_over_wf / potential >= self.grids['col'] \
                        and hv_over_hf / potential >= self.grids['row']:
                    break
            else:
                num_envs = self.grids
                if (floor(wv_over_wf / potential) *
                        floor(hv_over_hf / potential) >= num_envs):
                    break
            if potential == 0:
                raise ValueError()
        scale = potential

        assert scale != 0
        # else:
        num_rows = int(VIDEO_HEIGHT / floor(self.height * scale))
        num_cols = int(VIDEO_WIDTH / floor(self.width * scale))

        num_envs = num_rows * num_cols

        if specify_grids:
            assert num_rows >= self.grids['row']
            assert num_cols >= self.grids['col']
        self.num_rows = num_rows
        self.num_cols = num_cols
        frame_width = int(self.width * scale)
        frame_height = int(self.height * scale)

        assert num_rows * num_cols >= num_envs, \
            "row {}, col {}, envs {}".format(num_rows, num_cols, num_envs)
        assert num_cols * frame_width <= VIDEO_WIDTH
        assert num_rows * frame_height <= VIDEO_HEIGHT

        width_margin = (VIDEO_WIDTH - num_cols * frame_width) / (num_cols + 1)
        height_margin = (VIDEO_HEIGHT -
                         num_rows * frame_height) / (num_rows + 1)
        width_margin = int(width_margin)
        height_margin = int(height_margin)

        frame_range = []
        for i in range(num_envs):
            row_id = int(i / num_cols)
            col_id = int(i % num_cols)

            assert row_id < num_rows, (row_id, num_rows)
            assert col_id < num_cols, (col_id, num_cols)

            frame_range.append(
                {
                    "height": [
                        (height_margin + frame_height) * row_id +
                        height_margin + VIDEO_HEIGHT_EDGE,
                        (height_margin + frame_height) * (row_id + 1) +
                        VIDEO_HEIGHT_EDGE
                    ],
                    "width": [
                        (width_margin + frame_width) * col_id +
                        width_margin + VIDEO_WIDTH_EDGE,
                        (width_margin + frame_width) * (col_id + 1) +
                        VIDEO_WIDTH_EDGE
                    ],
                    "column":
                    col_id,
                    "row":
                    row_id,
                    "index":
                    i
                }
            )
        self.frame_range = frame_range
        self.scale = scale

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(
                self.path, frame.shape, self.frames_per_sec
            )
        try:
            self.encoder.capture_frame(frame)
        except error.InvalidFrame as e:
            logger.warn(
                'Tried to pass invalid video frame, marking as broken: %s', e
            )
            self.broken = True
        else:
            self.empty = False


class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise error.InvalidFrame(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), "
                "i.e., RGB values for a w-by-h image, with an optional alpha "
                "channel.".format(frame_shape)
            )
        self.wh = (w, h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        elif distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise error.DependencyNotInstalled(
                """Found neither the ffmpeg nor avconv executables. On OS X, 
                you can install ffmpeg via `brew install ffmpeg`. On most 
                Ubuntu variants, `sudo apt-get install ffmpeg` should do it. 
                On Ubuntu 14.04, however, you'll need to install avconv with 
                `sudo apt-get install libav-tools`."""
            )

        self.start()

    @property
    def version_info(self):
        return {
            'backend':
            self.backend,
            'version':
            str(
                subprocess.check_output(
                    [self.backend, '-version'], stderr=subprocess.STDOUT
                )
            ),
            'cmdline':
            self.cmdline
        }

    def start(self):
        self.cmdline = (
            self.backend,
            '-nostats',
            '-loglevel',
            'error',  # suppress warnings
            '-y',
            '-r',
            '%d' % self.frames_per_sec,
            # '-b', '2M',

            # input
            '-f',
            'rawvideo',
            '-s:v',
            '{}x{}'.format(*self.wh),
            '-pix_fmt',
            ('rgb32' if self.includes_alpha else 'rgb24'),
            '-i',
            '-',
            # this used to be /dev/stdin, which is not Windows-friendly

            # output
            '-vf',
            'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-vcodec',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-crf',
            '0',
            self.output_path
        )

        logger.debug('Starting ffmpeg with "%s"', ' '.join(self.cmdline))
        if hasattr(os, 'setsid'):  # setsid not present on Windows
            self.proc = subprocess.Popen(
                self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid
            )
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame(
                'Wrong type {} for {} (must be np.ndarray or np.generic)'.
                format(type(frame), frame)
            )
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame(
                "Your frame has shape {}, but the VideoRecorder is "
                "configured for shape {}.".format(
                    frame.shape, self.frame_shape
                )
            )
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                "Your frame has data type {}, but we require uint8 (i.e. RGB "
                "values from 0-255).".format(frame.dtype)
            )

        if distutils.version.LooseVersion(
                np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error(
                "VideoRecorder encoder exited with status {}".format(ret)
            )


def restore_agent(run_name, ckpt, env_name, config=None):
    cls = get_agent_class(run_name)
    if config is None:
        config = build_config(ckpt, {"num_gpus_per_worker": 0.1})
    # This is a workaround
    if run_name == "ES":
        config["num_workers"] = 1
    agent = cls(env=env_name, config=config)
    if ckpt is not None:
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        agent.restore(ckpt)
    return agent


def initialize_ray(local_mode=False, num_gpus=0, test_mode=False):
    if not ray.is_initialized():
        ray.init(
            logging_level=logging.ERROR if not test_mode else logging.INFO,
            log_to_driver=test_mode,
            local_mode=local_mode,
            num_gpus=num_gpus
        )
        print("Sucessfully initialize Ray!")
    print("Available resources: ", ray.available_resources())

def get_random_string():
    return str(uuid.uuid4())[:8]


def _get_num_iters_from_ckpt_name(ckpt):
    base_name = os.path.basename(ckpt)
    assert "-" in base_name
    assert base_name.startswith("checkpoint")
    num_iters = eval(base_name.split("-")[1])
    assert isinstance(num_iters, int)
    return num_iters
