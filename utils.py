"""
Record video given a trained PPO model.

Usage:
    python record_video.py /YOUR_HOME/ray_results/EXP_NAME/TRAIL_NAME \
    -l 3000 --scene split -rf REWARD_FUNCTION_NAME
"""

from __future__ import absolute_import, division, print_function

import collections
import distutils
import json
import os
import subprocess
import tempfile
from math import floor

import cv2
import numpy as np
from gym import logger, error

# from gym.wrappers.monitoring.video_recorder import ImageEncoder

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


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
            self, base_path, FPS=50
    ):

        # modes = env.metadata.get('render.modes', [])
        # self._async = env.metadata.get('semantics.async')
        self.frame_shape = None
        # self.num_envs = env.num_envs
        self.frame_range = None
        self.scale = None

        self.ansi_mode = False
        # if 'rgb_array' not in modes:
        #     if 'ansi' in modes:
        #         self.ansi_mode = True
        #     else:
        #         logger.info(
        #             'Disabling video recorder because {} neither supports '
        #             'video mode "rgb_array" nor "ansi".'.format(env)
        #         )
        #         return


        self.last_frame = None
        # self.env = env

        # required_ext = '.json' if self.ansi_mode else '.mp4'
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
        # self.frames_per_sec = env.metadata.get('video.frames_per_second', 30)
        self.frames_per_sec = FPS
        self.encoder = None  # lazily start the process
        # self.broken = False

        # Dump metadata
        # self.metadata = metadata or {}
        # self.metadata['content_type'] = 'video/vnd.openai.ansivid' \
        #     if self.ansi_mode else 'video/mp4'
        # self.metadata_path = '{}.meta.json'.format(path_base)
        # self.write_metadata()

        logger.info('Starting new video recorder writing to %s', self.path)
        # self.empty = True
        self.initialized = False

    def put_text(self, img, text, pos, thickness=1):
        if not text:
            return
        cv2.putText(
            img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.scale + 0.15,
            (0, 0, 0), thickness
            # , cv2.LINE_AA # Unable the anti-aliasing
        )

    def _build_background(self, frames_dict):

        assert self.frames_per_sec is not None

        self.extra_num_frames = 5 * int(self.frames_per_sec)

        video_length = max([len(frames) for frames in frames_dict.values()])\
                       + self.extra_num_frames

        self.background = np.zeros(
            (video_length, VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype='uint8')

        self._add_fixed_text()

    def _add_fixed_text(self):
        # TODO can add title and names of each row or column.
        pass

    def build_grids(self, frames_dict):
        # background = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype='uint8')

        for rang, (title, frames) in zip(self.frame_range,
                                         frames_dict.items()):
            # for rang, (title, frame) in zip(self.frame_range,
            # frames.items()):

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
                frames = [cv2.resize(
                    frame, (int(self.width * self.scale),
                            int(self.height * self.scale)
                            ),
                    interpolation=cv2.INTER_CUBIC
                ) for frame in frames]
                frames = np.concatenate(frames)

            self.background[:len(frames), height[0]:height[1],
            width[0]:width[1], 2::-1] = frames

            # filled the extra number of frames
            self.background[len(frames):len(frames)+self.extra_num_frames,
            height[0]:height[1], width[0]:width[1], 2::-1] = frames[-1]

        return self.background

    def generate_video(self, frames_dict):
        """Render the given `env` and add the resulting frame to the video."""

        # assert isinstance(frames, list)

        logger.debug('Capturing video frame: path=%s', self.path)

        # render_mode = 'ansi' if self.ansi_mode else 'rgb_array'

        # frames = []
        # for env in self.envs:
        #     single_frame = env.render(mode=render_mode)
        #     frames.append(single_frame)

        # frames = self.env.render()

        if not self.initialized:
            tmp_frame = list(frames_dict.values())[0][0]
            self.width = tmp_frame.shape[1]
            self.height = tmp_frame.shape[0]
            self._build_frame_range()
            self.initialized = True

        self.build_grids(frames_dict)

        for frame in self.background:
            self.last_frame = frame
            self._encode_image_frame(frame)

        self._close()

    def _close(self):
        """Make sure to manually close, or else you'll leak the encoder
        process"""

        # Add extra 5 seconds static frames to help visualization.
        # for _ in range(5 * int(self.frames_per_sec)):
        #     self._encode_image_frame(self.last_frame)

        if self.encoder:
            logger.debug('Closing video encoder: path=%s', self.path)
            self.encoder.close()
            self.encoder = None
        else:
            # No frames captured. Set metadata, and remove the empty output
            # file.
            os.remove(self.path)

            # if self.metadata is None:
            #     self.metadata = {}
            # self.metadata['empty'] = True

        # If broken, get rid of the output file, otherwise we'd leak it.
        # if self.broken:
        #     logger.info(
        #         'Cleaning up paths for broken video recorder: path=%s '
        #         'metadata_path=%s', self.path, self.metadata_path
        #     )
        #
        #     # Might have crashed before even starting the output file,
        #     # don't try to remove in that case.
        #     if os.path.exists(self.path):
        #         os.remove(self.path)
        #
        #     if self.metadata is None:
        #         self.metadata = {}
        #     self.metadata['broken'] = True

        # self.write_metadata()

    # def write_metadata(self):
    #     with open(self.metadata_path, 'w') as f:
    #         json.dump(self.metadata, f)

    def _build_frame_range(self):
        def center_range(center, rang):
            return [int(center - rang / 2), int(center + range / 2)]

        # scale = sqrt(self.num_envs / (
        # floor(VIDEO_HEIGHT / self.height) * floor(VIDEO_WIDTH / self.width)))

        wv_over_wf = VIDEO_WIDTH / self.width
        hv_over_hf = VIDEO_HEIGHT / self.height
        for potential in np.arange(1, 0, -0.01):
            # potential = 1, 0.9, ...
            if floor(wv_over_wf / potential) * floor(hv_over_hf / potential
                                                     ) >= self.num_envs:
                break
            if potential == 0:
                raise ValueError()
        scale = potential

        assert scale != 0
        num_rows = int(VIDEO_HEIGHT / floor(self.height * scale))
        num_cols = int(VIDEO_WIDTH / floor(self.width * scale))
        frame_width = int(self.width * scale)
        frame_height = int(self.height * scale)

        assert num_rows * num_cols >= self.num_envs
        assert num_cols * frame_width <= VIDEO_WIDTH
        assert num_rows * frame_height <= VIDEO_HEIGHT

        width_margin = (VIDEO_WIDTH - num_cols * frame_width) / (num_cols + 1)
        height_margin = (VIDEO_HEIGHT -
                         num_rows * frame_height) / (num_rows + 1)
        width_margin = int(width_margin)
        height_margin = int(height_margin)

        frame_range = []
        for i in range(self.num_envs):
            row_id = int(i / num_cols)
            col_id = int(i % num_cols)

            assert row_id < num_rows, (row_id, num_rows)
            assert col_id < num_cols, (col_id, num_cols)

            frame_range.append(
                {
                    "height": [
                        (height_margin + frame_height) * row_id +
                        height_margin,
                        (height_margin + frame_height) * (row_id + 1)
                    ],
                    "width": [
                        (width_margin + frame_width) * col_id + width_margin,
                        (width_margin + frame_width) * (col_id + 1)
                    ]
                }
            )
        self.frame_range = frame_range
        self.scale = scale

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(
                self.path, frame.shape, self.frames_per_sec
            )
            self.metadata['encoder_version'] = self.encoder.version_info

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
                "channel."
                    .format(frame_shape)
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
                "configured for shape {}."
                    .format(frame.shape, self.frame_shape)
            )
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                "Your frame has data type {}, but we require uint8 (i.e. RGB "
                "values from 0-255)."
                    .format(frame.dtype)
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
