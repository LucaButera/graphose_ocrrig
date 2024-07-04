import math
from typing import Optional, Sequence, Union

import numpy as np
from matplotlib.colors import to_rgba_array
from numpy.random import Generator
from skimage.draw import ellipse, polygon
from skimage.morphology import convex_hull_image

from utils.data import graphose_rng

COLORS = [
    "yellow",
    "red",
    "green",
    "blue",
    "orange",
    "pink",
    "purple",
    "olive",
    "brown",
    "cyan",
]

BG_COLOR = (0, 0, 0, 255)


def bg_img(radius: int) -> np.array:
    return np.zeros((radius * 2, radius * 2, 4), dtype=np.uint8)


def str_to_rgb(c: str) -> np.array:
    return (to_rgba_array(c).squeeze() * 255).astype(np.uint8)


def random_rgb(rng: Generator = graphose_rng) -> np.array:
    return str_to_rgb(rng.choice(COLORS))


def move(p: np.array, d: float, a: float) -> np.array:
    return p + np.array((d * math.cos(a), d * math.sin(a)))


def thick_line(
    p1: np.ndarray, p2: np.ndarray, thickness: float = 2
) -> tuple[np.ndarray, np.ndarray]:
    normal = np.arctan2(*(p2 - p1)[::-1]) + np.pi / 2
    normal = np.arctan2(np.sin(normal), np.cos(normal))
    c = np.floor(
        np.array(
            (
                p1[0] + np.cos(normal) * thickness / 2,
                p2[0] + np.cos(normal) * thickness / 2,
                p2[0] - np.cos(normal) * thickness / 2,
                p1[0] - np.cos(normal) * thickness / 2,
            )
        )
    ).astype(np.intc)
    r = np.floor(
        np.array(
            (
                p1[1] + np.sin(normal) * thickness / 2,
                p2[1] + np.sin(normal) * thickness / 2,
                p2[1] - np.sin(normal) * thickness / 2,
                p1[1] - np.sin(normal) * thickness / 2,
            )
        )
    ).astype(np.intc)
    rr, cc = polygon(r, c)
    return rr, cc


def half_ellipse(
    img: np.array,
    center: np.array,
    r_radius: float,
    c_radius: float,
    angle: float,
    color: tuple[int, int, int, int],
    mirror: bool = False,
) -> np.array:
    # Draw a vertical ellipse inside a half dimensioned rectangle
    rr, cc = ellipse(
        r=0 if mirror else r_radius,
        c=c_radius,
        r_radius=r_radius,
        c_radius=c_radius,
        shape=(r_radius, 2 * c_radius),
    )
    ll = np.vstack((cc, rr))
    # Rotate by the required angle and translate to center
    rotate = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    translate = np.array((c_radius, 0 if mirror else r_radius))[..., None]
    ll = rotate @ (ll - translate) + center[..., None]
    # Use convex hull to avoid aliasing artifacts
    cc, rr = np.floor(ll).astype(np.intc)
    rr = rr.clip(0, img.shape[0] - 1)
    cc = cc.clip(0, img.shape[1] - 1)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[rr, cc] = 255
    mask = convex_hull_image(mask)
    img[mask] = color
    return img


def hollow_ellipse(
    img: np.array,
    center: np.array,
    r_radius: float,
    c_radius: float,
    color: tuple[int, int, int, int],
    angle: float = 0,
    bg: Optional[tuple[int, int, int, int]] = None,
    thickness: float = 0,
) -> np.array:
    if bg is None:
        bg = BG_COLOR
    # Move from [0, 2pi] to [-pi, pi]
    # from clockwise to counter-clockwise
    angle = angle % (2 * math.pi)
    if angle > math.pi:
        angle = math.pi - angle
    else:
        angle = -angle
    rr, cc = ellipse(
        r=center[1],
        c=center[0],
        r_radius=r_radius,
        c_radius=c_radius,
        rotation=angle,
    )
    img[rr, cc] = color
    if thickness > 0:
        rr, cc = ellipse(
            r=center[1],
            c=center[0],
            r_radius=r_radius - thickness,
            c_radius=c_radius - thickness,
            rotation=angle,
        )
        img[rr, cc] = bg
    return img


def finger(
    img: np.array,
    joints: np.array,
    thickness: float,
    color: tuple[int, int, int, int],
    indent: float,
) -> np.array:
    for i in range(joints.shape[0] - 1):
        j1, j2 = joints[i], joints[i + 1]
        center = (j1 + j2) / 2
        r_radius = thickness
        c_radius = np.linalg.norm(j1 - center)
        c_radius += c_radius * indent
        angle = np.arctan2(*(j1 - j2)[::-1])
        img = hollow_ellipse(
            img=img,
            center=center,
            r_radius=r_radius,
            c_radius=c_radius,
            angle=angle,
            color=color,
        )
    return img


def chain(
    img: np.array,
    joints: np.array,
    thickness: float,
    color: tuple[int, int, int, int],
):
    for i in range(joints.shape[0] - 1):
        j1, j2 = joints[i], joints[i + 1]
        rr, cc = thick_line(j1, j2, thickness=thickness)
        img[rr, cc] = color
        img = hollow_ellipse(
            img=img,
            center=j1,
            r_radius=thickness / 2,
            c_radius=thickness / 2,
            color=color,
        )
        img = hollow_ellipse(
            img=img,
            center=j2,
            r_radius=thickness / 2,
            c_radius=thickness / 2,
            color=color,
        )
    return img


def trapezoid(
    img: np.ndarray,
    b_center: np.array,
    b: float,
    t: float,
    h: float,
    a: float,
    color: tuple[int, int, int, int],
) -> np.array:
    t_center = move(b_center, h, a + math.pi / 2)
    points = np.array(
        [
            move(b_center, b / 2, a),
            move(b_center, -b / 2, a),
            move(t_center, -t / 2, a),
            move(t_center, t / 2, a),
        ]
    )
    rr, cc = polygon(points[:, 1], points[:, 0])
    img[rr, cc] = color
    return img


def rectangle(
    img: np.ndarray,
    center: np.array,
    l1: float,
    l2: float,
    a: float,
    color: tuple[int, int, int, int],
) -> np.array:
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    points = np.array(
        [
            center + rotation @ np.array((l1 / 2, l2 / 2)).T,
            center + rotation @ np.array((-l1 / 2, l2 / 2)).T,
            center + rotation @ np.array((-l1 / 2, -l2 / 2)).T,
            center + rotation @ np.array((l1 / 2, -l2 / 2)).T,
        ]
    )
    rr, cc = polygon(points[:, 1], points[:, 0])
    img[rr, cc] = color
    return img


def segment_chain(
    start: np.array,
    base_angle: float,
    angles: Union[Sequence[float], float],
    n_seg: int,
    seg_lens: Union[Sequence[float], float],
) -> np.array:
    """
    Creates a sequence of joints representing chained segments.
    :param start: The initial position as a 2D coordinate.
    :param base_angle: The initial angle in radians.
    :param angles: Sequence of angles used to orient segments. If the length is
        less than the number of segments the last value will be reused.
    :param n_seg: Number of segments.
    :param seg_lens: Sequence of values used as segments length. If the length is
        less than the number of segments the last value will be reused.
    :return: A sequence of 2D coordinates
    """
    angles, seg_lens = np.array(angles), np.array(seg_lens)
    assert angles.ndim == 1 or angles.size == 1
    joints = [start]
    cum_angle = base_angle
    for i in range(n_seg):
        if seg_lens.size == 1:
            curr_len = seg_lens
        elif i < seg_lens.size - 1:
            curr_len = seg_lens[i]
        else:
            curr_len = seg_lens[-1]
        if i < angles.size - 1:
            joints.append(move(joints[-1], curr_len, cum_angle + angles[i]))
        else:
            curr_angle = angles[-1] if angles.size > 1 else angles
            joints.append(move(joints[-1], curr_len, cum_angle + curr_angle))
            cum_angle += curr_angle
    return np.array(joints)
