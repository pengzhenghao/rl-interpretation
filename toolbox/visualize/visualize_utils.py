from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function
# import sys
# sys.path.append("../")

# import collections
import distutils
import logging
import os
# import pickle
import subprocess
import tempfile
import time
# import uuid
from math import floor

import cv2
import numpy as np
import ray
from PIL import Image
from gym import logger, error
# from gym.envs.box2d import BipedalWalker
# from ray.rllib.agents.registry import get_agent_class
# from ray.tune.util import merge_dicts

ORIGINAL_VIDEO_WIDTH = 1920
ORIGINAL_VIDEO_HEIGHT = 1080

VIDEO_WIDTH_EDGE = 100
VIDEO_HEIGHT_EDGE = 20
# VIDEO_WIDTH_EDGE = 0
# VIDEO_HEIGHT_EDGE = 0

VIDEO_WIDTH = ORIGINAL_VIDEO_WIDTH - 2 * VIDEO_WIDTH_EDGE
VIDEO_HEIGHT = ORIGINAL_VIDEO_HEIGHT - 2 * VIDEO_HEIGHT_EDGE


def touch(path):
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    open(path, 'a').close()


@ray.remote
def remote_generate_gif(frames, path, fps=50):
    # print("Enter remote_generate_gif!!!!")
    _generate_gif(frames, path, fps)
    # print("Exit remote_generate_gif!!!!")
    return 1


def local_generate_gif(frames, path, fps=50):
    _generate_gif(frames, path, fps)
    return None


def _generate_gif(frames, path, fps=50):
    assert isinstance(frames, np.ndarray)
    assert frames.dtype == 'uint8'
    assert frames.ndim == 4
    assert path.endswith(".gif")
    print("Current dir: {}, store path: {}".format(os.getcwd(), path))
    duration = int(1 / fps * 1000)
    images = [Image.fromarray(frame) for frame in frames]

    # work around to crop the frame
    # images = [Image.fromarray(frame[50:150, 50:150, :]) for frame in frames]
    # print("LOOK SHAPE", frames[0].shape)
    # print("LOOK SIZE", images[0].size)
    # org_width, org_height = images[0].size
    # scale = max(org_height, org_width) / 100
    # images = [img.resize((int(org_width / scale), int(org_height / scale)))
    #        for img in images]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )


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
    allow_gif_mode = [
        'clip', 'full', 'beginning', 'end', 'period', '3period', "hd"
    ]

    def __init__(
            # self, env, path=None, metadata=None, base_path=None
            self,
            base_path,
            grids=None,
            generate_gif=False,
            # gif_mode=None,
            fps=50,
            scale=None,
            test_mode=False
    ):
        # self.grids = int | dict
        # if int, then it represent the number of videos
        # if dict, then it should be like {agent_name: (row, col)}
        self.frame_shape = None
        assert generate_gif or isinstance(grids,
                                          int) or isinstance(grids, dict)
        self.grids = grids
        self.frame_range = None
        self.scale = scale or 1
        self.last_frame = None
        self.num_cols = None
        self.num_rows = None
        self.generate_gif = generate_gif
        # if self.generate_gif:
        # assert gif_mode in self.allow_gif_mode
        # self.gif_mode = gif_mode

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
        self.base_path = base_path
        self.test_mode = test_mode

        # path_base, actual_ext = os.path.splitext(self.path)

        # if actual_ext != required_ext:
        #     hint = " HINT: The environment is text-only, therefore we're " \
        #            "recording its text output in a structured JSON format." \
        #         if self.ansi_mode else ''
        #     raise error.Error(
        #         "Invalid path given: {} -- must have file "
        #         "extension {}.{}".format(self.path, required_ext, hint)
        #     )
        # Touch the file in any case, so we know it's present. (This
        # corrects for platform platform differences. Using ffmpeg on
        # OS X, the file is precreated, but not on Linux.
        if not generate_gif:
            touch(self.path)
        else:
            os.makedirs(self.base_path, exist_ok=True)

        self.extra_num_frames = None
        self.frames_per_sec = fps
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
            canvas=None,
            rotate=False,
            not_scale=False
    ):
        # print("put text {} at pos {} at time {}".format(text, pos, timestep))
        canvas = self.background if canvas is None else canvas
        if not text:
            return
        if timestep is None:
            timestep = list(range(len(canvas)))
        elif isinstance(timestep, int):
            timestep = [timestep]
        assert isinstance(timestep, list)

        if self.test_mode:
            timestep = [0]

        if not rotate:
            for t in timestep:
                cv2.putText(
                    canvas[t], text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    1 if not_scale else 0.38 * self.scale + 0.1, color,
                    thickness
                )
        else:
            text_img = np.zeros(
                (canvas[0].shape[1], canvas[0].shape[0], 4), dtype='uint8'
            )
            new_pos = ORIGINAL_VIDEO_HEIGHT - pos[1], pos[0]
            cv2.putText(
                text_img, text, new_pos, cv2.FONT_HERSHEY_SIMPLEX,
                1 if not_scale else 0.5 * self.scale + 0.15, color, thickness
            )
            # Only work for black background which is all zeros
            for t in timestep:
                canvas[t] += text_img[:, ::-1, :].swapaxes(0, 1)

    def _build_background(self, frames_dict):
        # self.background is a numpy array with shape
        # [video len, height, width, 4]
        assert self.frames_per_sec is not None
        self.extra_num_frames = 5 * int(self.frames_per_sec)
        video_length = max(
            [
                len(frames_info['frames'])
                for frames_info in frames_dict.values()
            ]
        ) + self.extra_num_frames

        if self.test_mode:
            video_length = 1

        self.background = np.zeros(
            (video_length, ORIGINAL_VIDEO_HEIGHT, ORIGINAL_VIDEO_WIDTH, 3),
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

    def _build_grid_of_frames(
            self, frames_dict, extra_info_dict, require_text
    ):
        if require_text:
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

            height = rang["height"]
            width = rang["width"]

            if self.scale != 1:
                interpolation = cv2.INTER_AREA if self.scale<1 \
                    else cv2.INTER_LINEAR
                frames = [
                    cv2.resize(
                        frame, (
                            int(self.width * self.scale
                                ), int(self.height * self.scale)
                        ),
                        interpolation=interpolation
                    ) for frame in frames
                ]
                frames = np.stack(frames)
            if self.test_mode:
                self.background[0, \
                height[0]:height[1], width[0]: width[1],
                2::-1] = frames[0]
            else:
                self.background[:len(frames), \
                height[0]:height[1], width[0]: width[1],
                2::-1] = frames

            # filled the extra number of frames
            if self.test_mode:
                pass
            else:
                self.background[len(frames):, height[0]:height[1], width[0]:
                            width[1], 2::-1] = frames[-1]
            if require_text:
                for information in extra_info_dict.values():
                    if 'pos_ratio' not in information:
                        continue
                    pos = self._get_pos(
                        *information['pos_ratio'], width, height
                    )
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

    def generate_video(self, frames_dict, extra_info_dict, require_text=True):
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

        if self.generate_gif:
            # self.scale = 1
            name_path_dict = self._generate_gif(frames_dict, extra_info_dict)
            return name_path_dict
            # return self.base_path

        if not self.initialized:
            info = extra_info_dict['frame_info']
            # tmp_frame = list(frames_dict.values())[0][0]
            self.width = info['width']
            self.height = info['height']
            self._build_frame_range()
            self.initialized = True

        self._build_background(frames_dict)

        self._build_grid_of_frames(frames_dict, extra_info_dict, require_text)
        if self.test_mode:
            return self.background[0]

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

    @staticmethod
    def _get_pos(left_ratio, bottom_ratio, width, height):
        return (
            int(left_ratio * width[0] + (1 - left_ratio) * width[1]),
            int(bottom_ratio * height[0] + (1 - bottom_ratio) * height[1])
        )

    def _generate_gif(self, frames_dict, extra_info_dict):
        # self._add_things_on_backgaround(frames_dict, extra_info_dict)
        obj_list = []
        name_path_dict = {}
        num_workers = 16
        for idx, (title, frames_info) in \
                enumerate(frames_dict.items()):
            frames = frames_info['frames']
            width = (0, extra_info_dict['frame_info']['width'])
            height = (0, extra_info_dict['frame_info']['width'])

            resize_frames = frames
            if self.scale < 1:
                resize_frames = [
                    cv2.resize(
                        frame, (
                            int(width[1] * self.scale
                                ), int(height[1] * self.scale)
                        ),
                        interpolation=cv2.INTER_AREA
                    ) for frame in frames
                ]
                resize_frames = np.stack(resize_frames)

            for information in extra_info_dict.values():
                if 'pos_ratio' not in information:
                    continue
                pos = self._get_pos(*information['pos_ratio'], width, height)
                value = information[title]
                if isinstance(value, list):
                    # filter out the empty list if any.
                    if not value:
                        continue
                    value_sequence = value
                    for timestep, value in enumerate(value_sequence):
                        text = information['text_function'](value)
                        if text == 'X':
                            print(timestep, value, text, frames.shape)
                        self._put_text(timestep, text, pos, canvas=frames)
                else:
                    text = information['text_function'](value)
                    self._put_text(None, text, pos, canvas=frames)

            obj_ids, mode_path_dict = self._generate_gif_for_clip(
                title, frames, resize_frames, frames_info
            )
            obj_list.extend(obj_ids)
            name_path_dict[title] = mode_path_dict
            if len(obj_list) >= num_workers:
                # print("get_obj")
                ray.get(obj_list)
                obj_list.clear()
        if obj_list:
            print("obj_list: ", obj_list)
            ray.get(obj_list)
        return name_path_dict

    def _generate_gif_for_clip(
            self, agent_name, frames, resize_frames, frames_info
    ):
        print(
            "Start generating gif for agent <{}>. "
            "We will use these modes: {}".format(
                agent_name, self.allow_gif_mode
            )
        )
        length = len(frames)
        one_clip_length = min(3 * self.frames_per_sec, length)
        obj_ids = []
        mode_path_dict = {}
        for mode in self.allow_gif_mode:
            if mode == 'hd':
                clip = frames
                fps = self.frames_per_sec

            elif mode == 'clip':
                begin = resize_frames[:one_clip_length]
                end = resize_frames[-one_clip_length:]
                center = resize_frames[int((length - one_clip_length) / 2):
                                       int((length + one_clip_length) / 2)]
                clip = np.concatenate([begin, end, center])
                clip = clip[::2, ...]
                fps = self.frames_per_sec / 2

            elif mode == 'beginning':
                clip = resize_frames[:one_clip_length]
                clip = clip[::2, ...]
                fps = self.frames_per_sec / 2

            elif mode == 'end':
                clip = resize_frames[-one_clip_length:]
                clip = clip[::2, ...]
                fps = self.frames_per_sec / 2

            elif mode == 'period':
                period = min(frames_info['period'], length)
                clip = resize_frames[int((length - period) /
                                         2):int((length + period) / 2)]
                fps = self.frames_per_sec / 4

            elif mode == '3period':
                period = min(3 * frames_info['period'], length)
                clip = resize_frames[int((length - period) /
                                         2):int((length + period) / 2)]
                fps = self.frames_per_sec / 4

            else:
                continue

            # we consider the base_path is to .../exp_name/
            gif_path = os.path.join(
                self.base_path, agent_name.replace(" ", "-"),
                "{}_{}.gif".format(agent_name.replace(" ", "-"), mode)
            )
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            # print("input: ", gif_path, int(fps))
            obj_id = remote_generate_gif.remote(clip, gif_path, int(fps))
            print("Collect obj_id from remote_generate_gif: ", obj_id)
            obj_ids.append(obj_id)
            mode_path_dict[mode] = gif_path
        return obj_ids, mode_path_dict

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

        # For sunhao modification
        # search_range = [2, 1.5] + np.arange(1, 0, -0.01).tolist()
        # for potential in search_range:
        #     # potential = 1, 0.9, ...
        #     if specify_grids:
        #         num_envs = None
        #         if wv_over_wf / potential >= self.grids['col'] \
        #                 and hv_over_hf / potential >= self.grids['row']:
        #             break
        #     else:
        #         num_envs = self.grids
        #         if (floor(wv_over_wf / potential) *
        #                 floor(hv_over_hf / potential) >= num_envs):
        #             break
        #     if potential == 0:
        #         raise ValueError()
        # print("Sacle = ", potential)
        # scale = potential
        num_envs = None
        scale = 0.6

        assert scale != 0
        # else:

        num_cols = int(VIDEO_WIDTH / floor(self.width * scale))
        num_rows = int(VIDEO_HEIGHT / floor(self.height * scale))

        if num_envs is None:
            num_envs = num_rows * num_cols
        else:
            num_rows = min(num_rows, int(np.ceil(num_envs / num_cols)))

        # num_envs = num_rows * num_cols

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

        # width_margin = (VIDEO_WIDTH - num_cols * frame_width) / (num_cols + 1)
        # height_margin = (VIDEO_HEIGHT -
        #                  num_rows * frame_height) / (num_rows + 1)
        # width_margin = int(width_margin)
        # height_margin = int(height_margin)
        # Modified SUNHAO
        width_margin = height_margin = 8

        print("We use the width_margin: {}, height_margin: {}".format(
            width_margin, height_margin)
        )

        frame_range = []
        caption_offset = 0
        for i in range(num_envs):

            if i%12 == 0:
                caption_offset += 100

            row_id, col_id = self._get_location(i)

            assert row_id < num_rows, (row_id, num_rows)
            assert col_id < num_cols, (col_id, num_cols)

            frame_range.append(
                {
                    "height": [
                        (height_margin + frame_height) * row_id +
                        height_margin + VIDEO_HEIGHT_EDGE + caption_offset,
                        (height_margin + frame_height) * (row_id + 1) +
                        VIDEO_HEIGHT_EDGE + caption_offset
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

class SunhaoVideoRecorder(VideoRecorder):


    def _build_frame_range(self):
        specify_grids = not isinstance(self.grids, int)

        # if not specify_grids:
        wv_over_wf = VIDEO_WIDTH / self.width
        hv_over_hf = VIDEO_HEIGHT / self.height

        search_range = [2, 1.5] + np.arange(1, 0, -0.01).tolist()
        for potential in search_range:
            # potential = 1, 0.9, ...
            if specify_grids:
                num_envs = None
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
        print("Sacle = ", potential)
        scale = potential

        assert scale != 0
        # else:

        num_cols = int(VIDEO_WIDTH / floor(self.width * scale))
        num_rows = int(VIDEO_HEIGHT / floor(self.height * scale))

        if num_envs is None:
            num_envs = num_rows * num_cols
        else:
            num_rows = min(num_rows, int(np.ceil(num_envs / num_cols)))

        # num_envs = num_rows * num_cols

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
        height_margin = (VIDEO_HEIGHT - num_rows * frame_height) / (num_rows + 1)
        width_margin = int(width_margin)
        height_margin = int(height_margin)

        frame_range = []
        for i in range(num_envs):

            if i % 10 == 0 and i != 0:
                pass
                # now it is the special margin.


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
            # '-crf',
            # '0',
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
