import cv2
import numpy as np
import copy
import uuid


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

    def set(self, pxls):  #unimplemented
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
