"""
Record video given a trained PPO model.

Usage:
    python record_video.py /YOUR_HOME/ray_results/EXP_NAME/TRAIL_NAME \
    -l 3000 --scene split -rf REWARD_FUNCTION_NAME
"""

from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import collections
import distutils
import os
import subprocess
import tempfile
from math import floor

import cv2
import numpy as np
from gym import logger, error
from ray.rllib.agents.registry import get_agent_class

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

from gym.envs.box2d.bipedal_walker import (
    BipedalWalker, VIEWPORT_H, VIEWPORT_W, SCALE, TERRAIN_HEIGHT, TERRAIN_STEP
)
from Box2D.b2 import circleShape

from opencv_wrappers import Surface
import time
import pickle
from ray.tune.util import merge_dicts

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


def build_config(ckpt, args_config):
    ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
    config = {"log_level": "ERROR"}
    # Load configuration from file
    config_dir = os.path.dirname(ckpt)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args_config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint "
                "dir "
                "or "
                "its parent directory."
            )
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(1, config["num_workers"])

    config["log_level"] = "ERROR"
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
            num_envs,
            FPS=50
    ):
        self.frame_shape = None
        self.num_envs = num_envs
        self.frame_range = None
        self.scale = None
        self.last_frame = None

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

    def _put_text(self, timestep, text, pos, thickness=1):
        # print("put text {} at pos {} at time {}".format(text, pos, timestep))
        if not text:
            return
        cv2.putText(
            self.background[timestep], text, pos, cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * self.scale + 0.15, (0, 0, 0), thickness
            # , cv2.LINE_AA # Unable the anti-aliasing
        )

    def _build_background(self, frames_dict):
        assert self.frames_per_sec is not None
        self.extra_num_frames = 5 * int(self.frames_per_sec)
        video_length = max([len(frames_info['frames'])
                            for frames_info in
                            frames_dict.values()]) + self.extra_num_frames
        self.background = np.zeros(
            (video_length, VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype='uint8'
        )
        self._add_things_on_backgaround(frames_dict)

    def _add_things_on_backgaround(self, frames_dict):
        # TODO can add title and names of each row or column.
        # We can add all row / col name here!!!!
        return self.background

    def _build_grid_of_frames(self, frames_dict, extra_info_dict):
        # background = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype='uint8')

        for rang, (title, frames_info) in zip(self.frame_range,
                                         frames_dict.items()):
            # TODO we can add async execution here
            height = rang["height"]
            width = rang["width"]

            frames = frames_info['frames']

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
                frames = np.concatenate(frames)

            self.background[:len(frames), height[0]:height[1], width[0]:
            width[1], 2::-1] = frames

            # filled the extra number of frames
            self.background[len(frames):len(frames) +
                                        self.extra_num_frames,
            height[0]:height[1],
            width[0]:width[1], 2::-1] = frames[-1]

            for information in extra_info_dict.values():
                if 'pos_ratio' not in information:
                    continue
                pos = get_pos(*information['pos_ratio'])
                value = information[title]
                if isinstance(value, list):
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
                    for timestep in range(len(self.background)):
                        self._put_text(timestep, text, pos)

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

    def _build_frame_range(self):
        def center_range(center, rang):
            return [int(center - rang / 2), int(center + rang / 2)]

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


class OpencvViewer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = Surface(height=height, width=width)
        self.translation = 0, 0
        self.scale = 1, 1
        self.frame = np.empty((height, width, 4), dtype=np.uint8)
        self.frame.fill(255)

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.translation = -left, -bottom
        self.scale = scalex, scaley

    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        raise NotImplementedError

    def translate(self, point):
        point1 = point[0] + self.translation[0], point[1] + \
                 self.translation[1]
        point2 = point1[0] * self.scale[0], point1[1] * \
                 self.scale[1]
        return self.height - point2[1], point2[0]

    def draw_polygon(self, v, filled=True, **attrs):
        v = [self.translate(p) for p in v]
        color = scale_color(attrs["color"])
        self.surface.polygon(v, color)

    def draw_polyline(self, v, **attrs):
        color = scale_color(attrs["color"])
        thickness = attrs['thickness'] if 'thickness' in attrs \
            else attrs['linewidth']
        for point1, point2 in zip(v[:-1], v[1:]):
            point1 = self.translate(tuple(point1))
            point2 = self.translate(tuple(point2))
            self.surface.line(point1, point2, color, thickness)

    def draw_line(self, start, end, **attrs):
        start = self.translate(start)
        end = self.translate(end)
        self.surface.line(start, end, **attrs)

    def render(self, return_rgb_array):
        self.frame.fill(255)
        # if not return_rgb_array:
        #     self.surface.display(1)
        frame = self.surface.raw_data()
        return frame[:, :, 2::-1]

    def close(self):
        del self.surface


def scale_color(color_in_1):
    return tuple(int(c * 255) for c in color_in_1)


def restore_agent(run_name, ckpt, env_name, config=None):
    cls = get_agent_class(run_name)
    if config is None:
        config = build_config(ckpt, {})
    ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
    agent = cls(env=env_name, config=config)
    agent.restore(ckpt)
    return agent


class BipedalWalkerWrapper(BipedalWalker):
    def render(self, mode='rgb_array'):
        # This function is almost identical to the original one but the
        # importing of pyglet is avoided.
        if self.viewer is None:
            self.viewer = OpencvViewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(
            self.scroll, VIEWPORT_W / SCALE + self.scroll, 0,
                         VIEWPORT_H / SCALE
        )

        self.viewer.draw_polygon(
            [
                (self.scroll, 0),
                (self.scroll + VIEWPORT_W / SCALE, 0),
                (self.scroll + VIEWPORT_W / SCALE, VIEWPORT_H / SCALE),
                (self.scroll, VIEWPORT_H / SCALE),
            ],
            color=(0.9, 0.9, 1.0)
        )
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2: continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE: continue
            self.viewer.draw_polygon(
                [(p[0] + self.scroll / 2, p[1]) for p in poly],
                color=(1, 1, 1)
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar
                                         ) else self.lidar[len(self.lidar) -
                                                           i - 1]
            self.viewer.draw_polyline(
                [l.p1, l.p2], color=(1, 0, 0), linewidth=1
            )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    raise NotImplementedError
                    # t = rendering.Transform(translation=trans*f.shape.pos)
                    # self.viewer.draw_circle(f.shape.radius, 30,
                    # color=obj.color1).add_attr(t)
                    # self.viewer.draw_circle(f.shape.radius, 30,
                    # color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(
                        path, color=obj.color2, linewidth=2
                    )

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / SCALE
        x = TERRAIN_STEP * 3
        self.viewer.draw_polyline(
            [(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2
        )
        f = [
            (x, flagy2), (x, flagy2 - 10 / SCALE),
            (x + 25 / SCALE, flagy2 - 5 / SCALE)
        ]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
