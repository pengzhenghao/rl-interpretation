from gym.envs.box2d.bipedal_walker import (
    BipedalWalker, VIEWPORT_H, VIEWPORT_W, SCALE, TERRAIN_HEIGHT, TERRAIN_STEP
)
from Box2D.b2 import circleShape

import cv2
import numpy as np
import copy
import uuid


class Surface(object):
    __slots__ = [
        'center', 'shape', 'mask_key', '_img', 'height', 'width', 'name'
    ]

    def __init__(
            self,
            center_x=0,
            center_y=0,
            height=1,
            width=1,
            mask_key=23,
            name=None
    ):
        self.center = int(center_x), int(center_y)
        self.shape = self.height, self.width = (int(height), int(width))
        self.mask_key = int(mask_key & 255)
        self.name = name or str(uuid.uuid4())
        self._img = np.zeros((height, width, 4), dtype='uint8')

    def zero(self):
        self.pixels_ref().fill(0)
        return self

    def fill(self, color):
        color_to_uint = lambda color: np.uint32(
            (color[0] << 16) | (color[1] << 8) | (color[2])
        )
        assert isinstance(color, tuple) and len(color) == 3
        self.pixels_ref().fill(color_to_uint(color) | (self.mask_key << 24))
        return self

    def blit(self, surface):
        center_x, center_y = int(surface.center[0]), int(surface.center[1])
        half_height, half_width = int(surface.height / 2
                                      ), int(surface.width / 2)
        left = min(half_width, center_y)
        right = min(half_width, self.width - center_y)
        top = min(half_height, center_x)
        down = min(half_height, self.height - center_x)
        mask = surface.get_mask()
        negative_mask = 1 - mask
        img_src = surface.pixels_ref()
        screen_img = self.pixels_ref()
        screen_img[(center_x - top):(center_x + down), (center_y - left):(center_y + right)] *= \
            negative_mask[(half_height - top):(half_height + down), (half_width - left):(half_width + right)]
        screen_img[(center_x - top):(center_x + down), (center_y - left):(center_y + right)] += \
            img_src[(half_height - top):(half_height + down), (half_width - left):(half_width + right)] \
            * mask[(half_height - top):(half_height + down), (half_width - left):(half_width + right)]
        return self

    def get_mask(self):
        mask = self._img[:, :, 3] == self.mask_key
        return mask.astype(np.uint32)

    def raw_data(self):
        return self._img

    def pixels_ref(self):
        shape = (self.height, self.width)
        pxls = self._img.view('uint32').reshape(shape)
        return pxls

    def pixels_cpy(self):
        return self.pixels_ref().copy()

    def copy(self):
        return copy.deepcopy(self)

    def polygon(self, points, color):
        color = (color[2], color[1], color[0], self.mask_key)
        points = np.array(points).astype(np.int).reshape((len(points), 2))
        points = np.flip(points, 1)
        cv2.fillConvexPoly(self._img, points, color)

    def circle(self, center, radius, color, thickness=1):
        color = (color[2], color[1], color[0], self.mask_key)
        center = (int(center[1]), int(center[0]))
        self._img = cv2.circle(
            self._img, center, radius, color, thickness=thickness
        )

    def line(self, p1, p2, color, thickness=1):
        color = (color[2], color[1], color[0], self.mask_key)
        p1 = (int(p1[1]), int(p1[0]))
        p2 = (int(p2[1]), int(p2[0]))
        self._img = cv2.line(self._img, p1, p2, color, thickness=thickness)

    def put_text(self, pos, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        pos = (int(pos[1]), int(pos[0]))
        font_size = 0.4
        color = (255, 255, 255)
        thickness = 1
        cv2.putText(self._img, text, pos, font, font_size, color, thickness)

    def display(self, time=0):
        cv2.imshow(self.name, self._img)
        key = cv2.waitKey(time) & 0xFF
        ESC = 27
        if key == ESC:
            cv2.destroyAllWindows()
            exit()
        return key


def scale_color(color_in_1):
    return tuple(int(c * 255) for c in color_in_1)


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
        if not return_rgb_array:
            self.surface.display(1)
        frame = self.surface.raw_data()
        return frame[:, :, 2::-1]

    def close(self):
        del self.surface


class BipedalWalkerWrapper(BipedalWalker):
    def render(self, mode='human'):
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
                    # self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    # self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
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


if __name__ == '__main__':
    """
    Usage: 
    1. Ask administrator to install xvfb
    2. run:
        xvfb-run -s "-screen 0 600x400x24" python test_render.py
        
    Use mode="human" to see the pop-up OpenCV window.
    Use mode="rgb_array" to get the (X, X, 4) ndarray.
    """
    env = BipedalWalkerWrapper()
    env.reset()
    cnt = 0
    while True:
        cnt += 1
        frame = env.render(mode='rgb_array')
        print(
            'Current Time Step: {}, frame Shape: {}'.format(cnt, frame.shape)
        )
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Done!")
            break
    env.close()
