from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import copy
import uuid

import cv2
import numpy as np
from Box2D.b2 import circleShape
from gym.envs.box2d.bipedal_walker import (
    BipedalWalker, VIEWPORT_H, VIEWPORT_W, SCALE, TERRAIN_HEIGHT, TERRAIN_STEP,
    FPS
)

ORIGINAL_VIDEO_WIDTH = 1920
ORIGINAL_VIDEO_HEIGHT = 1080

VIDEO_WIDTH_EDGE = 100
VIDEO_HEIGHT_EDGE = 60

VIDEO_WIDTH = ORIGINAL_VIDEO_WIDTH - 2 * VIDEO_WIDTH_EDGE
VIDEO_HEIGHT = ORIGINAL_VIDEO_HEIGHT - 2 * VIDEO_HEIGHT_EDGE

# VIEWPORT_H = 400
# VIEWPORT_W = 400


class Surface:
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

    def set(self, pxls):  # unimplemented
        shape = pxls.shape
        if len(shape) == 2:
            height, width = shape
            shape = (height, width, 4)
            pxls = pxls.copy().reshape(shape=shape)
        else:
            assert len(shape) == 3
            height, width, n_channels = shape
            assert n_channels == 4
            pxls = pxls.copy()

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
        screen_img[(center_x - top):(center_x + down),
        (center_y - left):(center_y + right)] *= \
            negative_mask[(half_height - top):(half_height + down),
            (half_width - left):(half_width + right)]
        screen_img[(center_x - top):(center_x + down),
        (center_y - left):(center_y + right)] += \
            img_src[(half_height - top):(half_height + down),
            (half_width - left):(half_width + right)] \
            * mask[(half_height - top):(half_height + down),
              (half_width - left):(half_width + right)]
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


class OpencvViewer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = Surface(height=height, width=width)
        self.translation = 0, 0
        self.scale = 1, 1
        # self.frame = np.empty((height, width, 4), dtype=np.uint8)
        # self.frame.fill(255)

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
        # self.frame.fill(255)
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
        'video.frames_per_second': FPS
    }

    def render(self, mode='cropped'):
        # This function is almost identical to the original one but the
        # importing of pyglet is avoided.
        crop = (mode == 'cropped') or (mode == 'human_cropped')
        scroll = self.scroll + LEFT_DISPLACEMENT / SCALE if crop else \
            self.scroll
        scroll_v = BOTTOM_DISPLACEMENT / SCALE if crop else 0
        viewport_w = VIEWPORT_W - LEFT_DISPLACEMENT - RIGHT_DISPLACEMENT if \
            crop else VIEWPORT_W
        viewport_h = VIEWPORT_H - TOP_DISPLACEMENT - BOTTOM_DISPLACEMENT if \
            crop else VIEWPORT_H

        if self.viewer is None:
            self.viewer = OpencvViewer(viewport_w, viewport_h)

        self.viewer.set_bounds(
            scroll, scroll + viewport_w / SCALE, scroll_v,
            scroll_v + viewport_h / SCALE
        )

        self.viewer.draw_polygon(
            [
                (scroll, scroll_v),
                (scroll + viewport_w / SCALE, scroll_v),
                (scroll + viewport_w / SCALE, viewport_h / SCALE + scroll_v),
                (scroll, viewport_h / SCALE + scroll_v),
            ],
            color=(0.9, 0.9, 1.0)
        )
        for poly, x1, x2 in self.cloud_poly:
            if x2 < scroll / 2: continue
            if x1 > scroll / 2 + viewport_w / SCALE: continue
            self.viewer.draw_polygon(
                [(p[0] + scroll / 2, p[1]) for p in poly], color=(1, 1, 1)
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

        return self.viewer.render(mode == 'human' or mode == 'human_cropped')





class InverseFFTRendered(object):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'cropped', 'human_cropped'],
        'video.frames_per_second': FPS
    }

    def __init__(self):

        # The variable need to be updated
        self.scroll = None
        self.viewer = None
        self.cloud_poly = None
        self.terrain_poly = None
        self.drawlist = None

        pass


    def obs_to_state(self):

        def step(self, action):
            # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this
            # to receive a bit of stability help
            control_speed = False  # Should be easier as well
            if control_speed:
                self.joints[0].motorSpeed = float(
                    SPEED_HIP * np.clip(action[0], -1, 1))
                self.joints[1].motorSpeed = float(
                    SPEED_KNEE * np.clip(action[1], -1, 1))
                self.joints[2].motorSpeed = float(
                    SPEED_HIP * np.clip(action[2], -1, 1))
                self.joints[3].motorSpeed = float(
                    SPEED_KNEE * np.clip(action[3], -1, 1))
            else:
                self.joints[0].motorSpeed = float(
                    SPEED_HIP * np.sign(action[0]))
                self.joints[0].maxMotorTorque = float(
                    MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
                self.joints[1].motorSpeed = float(
                    SPEED_KNEE * np.sign(action[1]))
                self.joints[1].maxMotorTorque = float(
                    MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
                self.joints[2].motorSpeed = float(
                    SPEED_HIP * np.sign(action[2]))
                self.joints[2].maxMotorTorque = float(
                    MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
                self.joints[3].motorSpeed = float(
                    SPEED_KNEE * np.sign(action[3]))
                self.joints[3].maxMotorTorque = float(
                    MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

            self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

            pos = self.hull.position
            vel = self.hull.linearVelocity

            for i in range(10):
                self.lidar[i].fraction = 1.0
                self.lidar[i].p1 = pos
                self.lidar[i].p2 = (
                    pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                    pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE)
                self.world.RayCast(self.lidar[i], self.lidar[i].p1,
                                   self.lidar[i].p2)

            state = [
                self.hull.angle,
                # Normal angles up to 0.5 here, but sure more is possible.
                2.0 * self.hull.angularVelocity / FPS,
                0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,
                # Normalized to get -1..1 range
                0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
                self.joints[0].angle,
                # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
                self.joints[0].speed / SPEED_HIP,
                self.joints[1].angle + 1.0,
                self.joints[1].speed / SPEED_KNEE,
                1.0 if self.legs[1].ground_contact else 0.0,
                self.joints[2].angle,
                self.joints[2].speed / SPEED_HIP,
                self.joints[3].angle + 1.0,
                self.joints[3].speed / SPEED_KNEE,
                1.0 if self.legs[3].ground_contact else 0.0
            ]
            state += [l.fraction for l in self.lidar]
            assert len(state) == 24

            self.scroll = pos.x - VIEWPORT_W / SCALE / 5

            # shaping = 130 * pos[
            #     0] / SCALE  # moving forward is a way to receive reward (normalized to get 300 on completion)
            # shaping -= 5.0 * abs(state[
            #                          0])  # keep head straight, other than that and falling, any behavior is unpunished

            # reward = 0
            # if self.prev_shaping is not None:
            #     reward = shaping - self.prev_shaping
            # self.prev_shaping = shaping

            # for a in action:
            #     reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
                # normalized to about -50.0 using heuristic, more optimal agent should spend less

            # done = False
            # if self.game_over or pos[0] < 0:
            #     reward = -100
            #     done = True
            # if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            #     done = True
            return np.array(state), reward, done, {}




    def render(self, mode='cropped'):
        # This function is almost identical to the original one but the
        # importing of pyglet is avoided.
        crop = (mode == 'cropped') or (mode == 'human_cropped')
        scroll = self.scroll + LEFT_DISPLACEMENT / SCALE if crop else \
            self.scroll
        scroll_v = BOTTOM_DISPLACEMENT / SCALE if crop else 0
        viewport_w = VIEWPORT_W - LEFT_DISPLACEMENT - RIGHT_DISPLACEMENT if \
            crop else VIEWPORT_W
        viewport_h = VIEWPORT_H - TOP_DISPLACEMENT - BOTTOM_DISPLACEMENT if \
            crop else VIEWPORT_H

        if self.viewer is None:
            self.viewer = OpencvViewer(viewport_w, viewport_h)

        self.viewer.set_bounds(
            scroll, scroll + viewport_w / SCALE, scroll_v,
            scroll_v + viewport_h / SCALE
        )

        self.viewer.draw_polygon(
            [
                (scroll, scroll_v),
                (scroll + viewport_w / SCALE, scroll_v),
                (scroll + viewport_w / SCALE, viewport_h / SCALE + scroll_v),
                (scroll, viewport_h / SCALE + scroll_v),
            ],
            color=(0.9, 0.9, 1.0)
        )
        for poly, x1, x2 in self.cloud_poly:
            if x2 < scroll / 2: continue
            if x1 > scroll / 2 + viewport_w / SCALE: continue
            self.viewer.draw_polygon(
                [(p[0] + scroll / 2, p[1]) for p in poly], color=(1, 1, 1)
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < scroll: continue
            if poly[0][0] > scroll + viewport_w / SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        # self.lidar_render = (self.lidar_render + 1) % 100
        # i = self.lidar_render
        # if i < 2 * len(self.lidar):
        #     l = self.lidar[i] if i < len(self.lidar) \
        #         else self.lidar[len(self.lidar) - i - 1]
        #     self.viewer.draw_polyline(
        #         [l.p1, l.p2], color=(1, 0, 0), linewidth=1
        #     )

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

        return self.viewer.render(mode == 'human' or mode == 'human_cropped')


if __name__ == '__main__':
    # this is the test codes
    env = BipedalWalkerWrapper()
    env.reset()
    while True:
        act = env.action_space.sample()
        _, _, done, _ = env.step(act)
        ret = env.render("human_cropped")
        print(
            "Return type: {}. Shape: {}".format(
                type(ret), ret.shape if isinstance(ret, np.ndarray) else None
            )
        )
        if done:
            env.reset()
