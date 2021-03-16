# Copyright (c) Open-MMLab. All rights reserved.
from enum import Enum

import numpy as np

from mmcv.utils import is_str

COLOR_LIST = [
    'red',
    'green',
    'blue',
    'cyan',
    'yellow',
    'magenta',
    'white',
    'black',
    'carmine',
    'ruby',
    'camellia',
    'rose',
    'mauve',
    'apricot',
    'sandbeige',
    'rown',
    'coffee',
    'marigod',
    'champange',
    'apple_green',
    'resh',
    'foliage',
    'azure',
    'sky',
    'saxe',
    'turquoise',
    'royal',
    'lapis',
    'salvia',
    'slate',
    'sapphire',
    'mineral',
    'midnight',
    'wisteria',
    'clematis',
    'lilac']

COLOR_VALUE_BGR = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
    (0, 0, 0),
    (64, 0, 215),
    (82, 8, 200),
    (111, 91, 220),
    (154, 134, 238),
    (192, 152, 225),
    (107, 169, 229),
    (202, 214, 236),
    (18, 59, 113),
    (35, 75, 106),
    (0, 171, 247),
    (209, 229, 235),
    (25, 189, 158),
    (107, 208, 169),
    (86, 162, 135),
    (230, 174, 34),
    (219, 212, 177),
    (205, 176, 139),
    (197, 164, 0),
    (162, 80, 30),
    (152, 64, 19),
    (175, 119, 91),
    (151, 121, 100),
    (137, 87, 0),
    (106, 66, 56),
    (58, 22, 4),
    (159, 91, 115),
    (203, 191, 216),
    (203, 161, 187),
]


class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
    carmine = (64, 0, 215)
    ruby = (82, 8, 200)
    camellia = (111, 91, 220)
    rose = (154, 134, 238)
    mauve = (192, 152, 225)
    apricot = (107, 169, 229)
    sandbeige = (202, 214, 236)
    rown = (18, 59, 113)
    coffee = (35, 75, 106)
    marigod = (0, 171, 247)
    champange = (209, 229, 235)
    apple_green = (25, 189, 158)
    resh = (107, 208, 169)
    foliage = (86, 162, 135)
    azure = (230, 174, 34)
    sky = (219, 212, 177)
    saxe = (205, 176, 139)
    turquoise = (197, 164, 0)
    royal = (162, 80, 30)
    lapis = (152, 64, 19)
    salvia = (175, 119, 91)
    slate = (151, 121, 100)
    sapphire = (137, 87, 0)
    mineral = (106, 66, 56)
    midnight = (58, 22, 4)
    wisteria = (159, 91, 115)
    clematis = (203, 191, 216)
    lilac = (203, 161, 187)


def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')