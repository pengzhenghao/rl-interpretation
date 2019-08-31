from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import numpy as np
from Box2D.b2 import circleShape
from gym.envs.box2d.bipedal_walker import (
    BipedalWalker, VIEWPORT_H, VIEWPORT_W, SCALE, TERRAIN_HEIGHT, TERRAIN_STEP,
FPS
)

from opencv_wrappers import Surface

ORIGINAL_VIDEO_WIDTH = 1920
ORIGINAL_VIDEO_HEIGHT = 1080

VIDEO_WIDTH_EDGE = 100
VIDEO_HEIGHT_EDGE = 60

VIDEO_WIDTH = ORIGINAL_VIDEO_WIDTH - 2 * VIDEO_WIDTH_EDGE
VIDEO_HEIGHT = ORIGINAL_VIDEO_HEIGHT - 2 * VIDEO_HEIGHT_EDGE

# VIEWPORT_H = 400
# VIEWPORT_W = 400

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
        point1 = point[0] + self.translation[0], \
                 point[1] + self.translation[1]
        point2 = point1[0] * self.scale[0], \
                 point1[1] * self.scale[1]
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

    def render(self, display):
        self.frame.fill(255)
        if display:
            self.surface.display(1)
        frame = self.surface.raw_data()
        return frame[:, :, 2::-1]

    def close(self):
        del self.surface


def scale_color(color_in_1):
    return tuple(int(c * 255) for c in color_in_1)


TOP_DISPLACEMENT = 150
BOTTOM_DISPLACEMENT = 50
LEFT_DISPLACEMENT = 20
RIGHT_DISPLACEMENT = 380
# VIEWPORT_H = VIEWPORT_H - TOP_DISPLACEMENT - BOTTOM_DISPLACEMENT
# VIEWPORT_W = VIEWPORT_W - LEFT_DISPLACEMENT - RIGHT_DISPLACEMENT

class BipedalWalkerWrapper(BipedalWalker):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'cropped', 'human_cropped'],
        'video.frames_per_second' : FPS
    }

    def render(self, mode='cropped'):
        # This function is almost identical to the original one but the
        # importing of pyglet is avoided.
        crop = (mode=='cropped') or (mode=='human_cropped')
        scroll = self.scroll + LEFT_DISPLACEMENT/SCALE if crop else self.scroll
        scroll_v = BOTTOM_DISPLACEMENT/SCALE if crop else 0
        viewport_w = VIEWPORT_W - LEFT_DISPLACEMENT - RIGHT_DISPLACEMENT if crop else VIEWPORT_W
        viewport_h = VIEWPORT_H - TOP_DISPLACEMENT - BOTTOM_DISPLACEMENT if crop else VIEWPORT_H

        if self.viewer is None:
            self.viewer = OpencvViewer(viewport_w, viewport_h)

        self.viewer.set_bounds(
            scroll,
            scroll + viewport_w / SCALE,
            scroll_v,
            scroll_v + viewport_h / SCALE
        )

        self.viewer.draw_polygon(
            [
                (scroll, scroll_v),
                (scroll+ viewport_w / SCALE, scroll_v),
                (scroll + viewport_w / SCALE, viewport_h / SCALE + scroll_v),
                (scroll, viewport_h / SCALE + scroll_v),
            ],
            color=(0.9, 0.9, 1.0)
        )
        for poly, x1, x2 in self.cloud_poly:
            if x2 < scroll / 2: continue
            if x1 > scroll / 2 + viewport_w / SCALE: continue
            self.viewer.draw_polygon(
                [(p[0] + scroll / 2, p[1]) for p in poly],
                color=(1, 1, 1)
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < scroll: continue
            if poly[0][0] > scroll + viewport_w / SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) \
                else self.lidar[len(self.lidar) - i - 1]
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

        return self.viewer.render(mode=='human' or mode=='human_cropped')

if __name__ == '__main__':
    # this is the test codes
    env = BipedalWalkerWrapper()
    env.reset()
    while True:
        act = env.action_space.sample()
        _, _, done, _ = env.step(act)
        ret = env.render("human_cropped")
        print("Return type: {}. Shape: {}".format(
            type(ret), ret.shape if isinstance(ret, np.ndarray) else None)
        )
        if done:
            env.reset()