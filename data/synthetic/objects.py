from abc import ABC, abstractmethod
from functools import cached_property

import networkx as nx
import numpy as np
from numpy.random import Generator
from PIL import Image, ImageDraw
from PIL.ImageDraw import _compute_regular_polygon_vertices
from scipy.stats._qmc import PoissonDisk

from data.synthetic.primitives import (
    bg_img,
    chain,
    finger,
    half_ellipse,
    hollow_ellipse,
    move,
    random_rgb,
    rectangle,
    segment_chain,
    thick_line,
    trapezoid,
)
from utils.data import graphose_rng


class DrawableGraph(ABC):
    def __init__(
        self, position: tuple[int, int], radius: int, rng: Generator = graphose_rng
    ):
        super().__init__()
        self.position = position
        self.radius = radius
        self.rng = rng

    @cached_property
    @abstractmethod
    def sprite(self) -> Image:
        pass

    @cached_property
    @abstractmethod
    def graph(self) -> nx.Graph:
        pass


class StickMan(DrawableGraph):
    pass


class Animal(DrawableGraph):
    pass


class Arm(DrawableGraph):
    def __init__(
        self, position: tuple[int, int], radius: int, rng: Generator = graphose_rng
    ):
        super().__init__(position, radius, rng)
        self.n_arm_segments = self.rng.integers(3, 8).item()
        self.n_prong_segments = 2

    @cached_property
    def graph(self) -> nx.Graph:
        rotation = np.radians(self.rng.uniform(0, 360)).item()
        center = np.array((self.radius, self.radius))
        arm = nx.Graph()
        seg_angles = np.radians(self.rng.uniform(-45, 45, self.n_arm_segments)).cumsum()
        seg_lens = (
            self.rng.uniform(0.8, 1, self.n_arm_segments)
            * self.radius
            / self.n_arm_segments
        )
        seg_color = random_rgb(self.rng)
        joints = segment_chain(
            center,
            rotation,
            seg_angles,
            self.n_arm_segments,
            seg_lens,
        )
        arm.add_nodes_from(
            [
                (f"arm_{i}", {"pos": joint, "color": seg_color, "type": "robot_arm"})
                for i, joint in enumerate(joints)
            ]
        )
        arm.add_edges_from(
            [(f"arm_{i}", f"arm_{i + 1}") for i in range(joints.shape[0] - 1)]
        )
        arm.add_node(
            "base",
            pos=move(center, self.radius / 8, rotation + np.pi),
            color=random_rgb(self.rng),
            type="robot_base",
        )
        arm.add_edge("base", "arm_0")
        prong_rotation = np.radians(self.rng.uniform(-30, 30)).item()
        prong_angle = np.radians(self.rng.uniform(60, 120)).item()
        inner_prong_angle = np.pi / 2 - np.radians(self.rng.uniform(0, 60)).item()
        prong_color = random_rgb(self.rng)
        prong1 = segment_chain(
            joints[-1],
            rotation + np.pi,
            np.array([prong_rotation + prong_angle / 2, -inner_prong_angle]).cumsum(),
            self.n_prong_segments,
            np.array([0.8, 0.6]) * self.radius / self.n_arm_segments,
        )[1:]
        prong2 = segment_chain(
            joints[-1],
            rotation + np.pi,
            np.array([prong_rotation - prong_angle / 2, inner_prong_angle]).cumsum(),
            self.n_prong_segments,
            np.array([0.8, 0.6]) * self.radius / self.n_arm_segments,
        )[1:]
        arm.add_nodes_from(
            [
                (
                    f"prong_{i}_{j}",
                    {"pos": prong[j], "color": prong_color, "type": "robot_prong"},
                )
                for i, prong in enumerate([prong1, prong2])
                for j in range(len(prong))
            ]
        )
        arm.add_edges_from(
            [
                (
                    f"prong_{i}_{j}",
                    f"arm_{self.n_arm_segments}" if j == 0 else f"prong_{i}_{j - 1}",
                )
                for i, prong in enumerate([prong1, prong2])
                for j in range(len(prong))
            ]
        )
        return arm

    @cached_property
    def sprite(self) -> Image:
        sprite = bg_img(self.radius)
        arm = self.graph
        sprite = chain(
            img=sprite,
            joints=np.array(
                [arm.nodes[f"arm_{i}"]["pos"] for i in range(self.n_arm_segments + 1)]
            ),
            thickness=self.radius / 12,
            color=arm.nodes["arm_0"]["color"],
        )
        sprite = rectangle(
            sprite,
            arm.nodes["base"]["pos"],
            self.radius / 4,
            self.radius / 2,
            np.arctan2(*(arm.nodes["base"]["pos"] - arm.nodes["arm_0"]["pos"])[::-1]),
            arm.nodes["base"]["color"],
        )
        for i in range(2):
            sprite = chain(
                img=sprite,
                joints=np.array(
                    [
                        arm.nodes[f"arm_{self.n_arm_segments}"]["pos"],
                        *(
                            arm.nodes[f"prong_{i}_{j}"]["pos"]
                            for j in range(self.n_prong_segments)
                        ),
                    ]
                ),
                thickness=self.radius / 18,
                color=arm.nodes[f"prong_{i}_0"]["color"],
            )
        return Image.fromarray(sprite, "RGBA")


class Hand(DrawableGraph):
    def __init__(
        self, position: tuple[int, int], radius: int, rng: Generator = graphose_rng
    ):
        super().__init__(position, radius, rng)
        self.n_phalanx = 3
        self.n_fingers = 4
        self.extent = self.radius * 2 / 3
        self.thickness = self.extent / 4
        self.indent = 1 / 3
        self.flipped = self.rng.uniform() > 0.5
        self.phalanx_factors = (0.7, 0.9, 1, 0.85)[:: (-1 if self.flipped else 1)]
        self.center = np.array((self.radius, self.radius))

    @cached_property
    def graph(self) -> nx.Graph:
        rotation = np.radians(self.rng.uniform(0, 360)).item()
        hand = nx.Graph()
        wrist = move(
            self.center,
            -6 / 7 * self.extent + self.thickness / self.n_phalanx,
            rotation,
        )
        for i in range(self.n_fingers):
            color = random_rgb(self.rng)
            joints = segment_chain(
                move(
                    self.center,
                    (i + (1 - self.n_fingers) / 2) * self.extent / self.n_fingers,
                    rotation + np.pi / 2,
                ),
                rotation,
                np.radians(
                    np.array(
                        [
                            self.rng.uniform(-20, 20),
                            self.rng.uniform(0, 15) * (-1 if self.flipped else 1),
                        ]
                    )
                ),
                self.n_phalanx,
                self.phalanx_factors[i] * self.extent / self.n_phalanx,
            )
            hand.add_nodes_from(
                [
                    (
                        f"finger_{i}_joint_{j}",
                        {"pos": joint, "color": color, "type": "hand_finger"},
                    )
                    for j, joint in enumerate(joints)
                ]
            )
            hand.add_edges_from(
                [
                    (f"finger_{i}_joint_{j}", f"finger_{i}_joint_{j + 1}")
                    for j in range(joints.shape[0] - 1)
                ]
            )
        thumb_angles = -np.radians(
            np.array([self.rng.uniform(0, 60), self.rng.uniform(0, 15)])
        ).cumsum()
        thumb = segment_chain(
            move(
                wrist,
                3 / 8 * self.extent * (-1 if self.flipped else 1),
                rotation + np.pi / 2,
            ),
            rotation + np.pi / 2 * (-1 if self.flipped else 1),
            thumb_angles * (-1 if self.flipped else 1),
            self.n_phalanx - 1,
            self.extent / self.n_phalanx,
        )
        thumb_color = random_rgb(self.rng)
        hand.add_nodes_from(
            [
                (
                    f"thumb_{i}",
                    {
                        "pos": joint,
                        "color": thumb_color,
                        "type": "hand_finger",
                    },
                )
                for i, joint in enumerate(thumb)
            ]
        )
        hand.add_edges_from(
            [(f"thumb_{i}", f"thumb_{i + 1}") for i in range(thumb.shape[0] - 1)]
        )
        hand.add_node("wrist", pos=wrist, color=random_rgb(self.rng), type="hand_wrist")
        hand.add_edges_from(
            [
                ("wrist", finger_start)
                for finger_start in [
                    "thumb_0",
                    *(f"finger_{i}_joint_0" for i in range(self.n_fingers)),
                ]
                if finger_start in hand
            ]
        )
        return hand

    @cached_property
    def sprite(self) -> Image:
        sprite = bg_img(self.radius)
        hand = self.graph
        for i in range(self.n_fingers):
            sprite = finger(
                img=sprite,
                joints=np.array(
                    [
                        hand.nodes[f"finger_{i}_joint_{j}"]["pos"]
                        for j in range(self.n_phalanx + 1)
                    ]
                ),
                thickness=self.thickness * self.phalanx_factors[i] / self.n_phalanx,
                color=hand.nodes[f"finger_{i}_joint_0"]["color"],
                indent=self.indent,
            )
        sprite = finger(
            img=sprite,
            joints=np.array(
                [hand.nodes[f"thumb_{i}"]["pos"] for i in range(self.n_phalanx)]
            ),
            thickness=self.thickness / self.n_phalanx,
            color=hand.nodes["thumb_0"]["color"],
            indent=self.indent,
        )
        sprite = trapezoid(
            sprite,
            self.center,
            self.extent,
            3 / 4 * self.extent,
            6 / 7 * self.extent,
            np.arctan2(*(self.center - hand.nodes["wrist"]["pos"])[::-1]) + np.pi / 2,
            hand.nodes["wrist"]["color"],
        )
        return Image.fromarray(sprite, "RGBA")


class Pie(DrawableGraph):
    @cached_property
    def graph(self) -> nx.Graph:
        rotation = np.radians(self.rng.uniform(0, 360)).item()
        mouth_angle = np.radians(self.rng.uniform(0, 135)).item()
        center = np.array((self.radius, self.radius))
        tip1 = move(center, self.radius, rotation + mouth_angle)
        tip2 = move(center, self.radius, rotation)
        color = random_rgb(self.rng)
        scissors = nx.Graph()
        scissors.add_nodes_from(
            [
                ("center", {"pos": center, "color": color, "type": "pie_center"}),
                ("tip1", {"pos": tip1, "color": color, "type": "pie_tip"}),
                ("tip2", {"pos": tip2, "color": color, "type": "pie_tip"}),
            ]
        )
        scissors.add_edges_from(
            [
                ("center", "tip1"),
                ("center", "tip2"),
            ]
        )
        return scissors

    @cached_property
    def sprite(self) -> Image:
        pie = self.graph
        sprite = Image.fromarray(bg_img(self.radius), "RGBA")
        draw = ImageDraw.Draw(sprite)
        draw.pieslice(
            xy=((0, 0), (self.radius * 2, self.radius * 2)),
            start=np.degrees(
                np.arctan2(
                    *(pie.nodes["tip1"]["pos"] - pie.nodes["center"]["pos"])[::-1]
                )
            ).item(),
            end=np.degrees(
                np.arctan2(
                    *(pie.nodes["tip2"]["pos"] - pie.nodes["center"]["pos"])[::-1]
                )
            ).item(),
            fill=tuple(pie.nodes["center"]["color"]),
        )
        return sprite


class Polygon(DrawableGraph):
    @cached_property
    def graph(self) -> nx.Graph:
        filled = self.rng.uniform() > 0.5
        n_sides = self.rng.integers(3, 8, endpoint=True).item()
        rotation = self.rng.uniform(0, 360)
        color = random_rgb(self.rng)
        points = np.array(
            _compute_regular_polygon_vertices(
                (self.radius, self.radius, self.radius), n_sides, rotation
            )
        )
        poly = nx.Graph()
        poly.add_nodes_from(
            [
                (i, {"pos": point, "color": color, "type": "polygon_vertex"})
                for i, point in enumerate(points)
            ]
        )
        poly.add_edges_from(
            [
                (i, i + 1 if i < points.shape[0] - 1 else 0)
                for i in range(points.shape[0])
            ]
        )
        if filled:
            poly.add_node(
                "center",
                pos=np.array((self.radius, self.radius)),
                color=color,
                type="polygon_center",
            )
            poly.add_edges_from([("center", i) for i in range(points.shape[0])])
        return poly

    @cached_property
    def sprite(self) -> Image:
        poly = self.graph
        filled = "center" in poly
        sprite = Image.fromarray(bg_img(self.radius), "RGBA")
        draw = ImageDraw.Draw(sprite)
        draw.polygon(
            xy=[tuple(poly.nodes[n]["pos"]) for n in poly if not n == "center"],
            fill=tuple(poly.nodes[0]["color"]) if filled else None,
            outline=None if filled else tuple(poly.nodes[0]["color"]),
            width=1 if filled else self.radius // 8,
        )
        return sprite


class Scissors(DrawableGraph):
    @cached_property
    def graph(self) -> nx.Graph:
        rotation = np.radians(self.rng.uniform(0, 360)).item()
        angle = np.radians(self.rng.uniform(30, 90)).item()
        center = np.array((self.radius, self.radius))
        tip1 = move(center, self.radius, rotation)
        tip2 = move(center, self.radius, rotation + angle)
        handle1 = move(center, self.radius / 3, rotation + np.pi)
        handle2 = move(center, self.radius / 3, rotation + angle + np.pi)
        blade_color = random_rgb(self.rng)
        handle_color = random_rgb(self.rng)
        scissors = nx.Graph()
        scissors.add_nodes_from(
            [
                (
                    "center",
                    {"pos": center, "color": blade_color, "type": "scissors_pivot"},
                ),
                ("tip1", {"pos": tip1, "color": blade_color, "type": "scissors_tip"}),
                ("tip2", {"pos": tip2, "color": blade_color, "type": "scissors_tip"}),
                (
                    "handle1",
                    {"pos": handle1, "color": handle_color, "type": "scissors_handle"},
                ),
                (
                    "handle2",
                    {"pos": handle2, "color": handle_color, "type": "scissors_handle"},
                ),
            ]
        )
        scissors.add_edges_from(
            [
                ("center", "tip1"),
                ("center", "tip2"),
                ("center", "handle1"),
                ("center", "handle2"),
            ]
        )
        return scissors

    @cached_property
    def sprite(self) -> Image:
        scissors = self.graph
        blade_thickness = max(4, self.rng.uniform(self.radius / 8, self.radius / 4))
        handle_thickness = blade_thickness / 2
        sprite = bg_img(self.radius)
        sprite = half_ellipse(
            img=sprite,
            center=(scissors.nodes["center"]["pos"] + scissors.nodes["tip1"]["pos"])
            / 2,
            r_radius=blade_thickness,
            c_radius=np.linalg.norm(
                scissors.nodes["center"]["pos"] - scissors.nodes["tip1"]["pos"]
            )
            / 2,
            angle=np.arctan2(
                *(scissors.nodes["tip1"]["pos"] - scissors.nodes["center"]["pos"])[::-1]
            ),
            color=scissors.nodes["tip1"]["color"],
            mirror=False,
        )
        sprite = half_ellipse(
            img=sprite,
            center=(scissors.nodes["center"]["pos"] + scissors.nodes["tip2"]["pos"])
            / 2,
            r_radius=blade_thickness,
            c_radius=np.linalg.norm(
                scissors.nodes["center"]["pos"] - scissors.nodes["tip2"]["pos"]
            )
            / 2,
            angle=np.arctan2(
                *(scissors.nodes["tip2"]["pos"] - scissors.nodes["center"]["pos"])[::-1]
            ),
            color=scissors.nodes["tip2"]["color"],
            mirror=True,
        )
        sprite = hollow_ellipse(
            img=sprite,
            center=scissors.nodes["handle1"]["pos"],
            r_radius=np.linalg.norm(
                scissors.nodes["center"]["pos"] - scissors.nodes["handle1"]["pos"]
            )
            / 2,
            c_radius=np.linalg.norm(
                scissors.nodes["center"]["pos"] - scissors.nodes["handle1"]["pos"]
            ),
            angle=np.arctan2(
                *(scissors.nodes["handle1"]["pos"] - scissors.nodes["center"]["pos"])[
                    ::-1
                ]
            ),
            thickness=handle_thickness,
            color=scissors.nodes["handle1"]["color"],
        )
        sprite = hollow_ellipse(
            img=sprite,
            center=scissors.nodes["handle2"]["pos"],
            r_radius=np.linalg.norm(
                scissors.nodes["center"]["pos"] - scissors.nodes["handle2"]["pos"]
            )
            / 2,
            c_radius=np.linalg.norm(
                scissors.nodes["center"]["pos"] - scissors.nodes["handle2"]["pos"]
            ),
            angle=np.arctan2(
                *(scissors.nodes["handle2"]["pos"] - scissors.nodes["center"]["pos"])[
                    ::-1
                ]
            ),
            thickness=handle_thickness,
            color=scissors.nodes["handle2"]["color"],
        )
        return Image.fromarray(sprite, "RGBA")


class Lattice(DrawableGraph):
    @cached_property
    def graph(self) -> nx.Graph:
        n_nodes = self.rng.integers(3, 9, endpoint=True).item()
        samples = PoissonDisk(d=2, radius=0.1, seed=self.rng).integers(
            l_bounds=(0, 0), u_bounds=(self.radius * 2, self.radius * 2), n=n_nodes
        )
        lattice = nx.empty_graph(samples.shape[0])
        nx.set_node_attributes(
            lattice, {k: random_rgb(self.rng) for k in lattice}, "color"
        )
        nx.set_node_attributes(lattice, {k: samples[k] for k in lattice}, "pos")
        nx.set_node_attributes(lattice, {k: "lattice_vertex" for k in lattice}, "type")
        lattice.add_edges_from(
            nx.geometric_edges(lattice, radius=0.2 * 2 * self.radius)
        )
        return lattice

    @cached_property
    def sprite(self) -> Image:
        thickness = self.radius // 12
        sprite = bg_img(self.radius)
        lattice = self.graph
        for edge in lattice.edges:
            n1, n2 = lattice.nodes[edge[0]], lattice.nodes[edge[1]]
            p1, p2 = n1["pos"], n2["pos"]
            rr, cc = thick_line(p1, p2, thickness=thickness)
            ll = np.stack([cc, rr], axis=-1)
            dd = np.linalg.norm(p1[None] - ll, axis=1)[..., None]
            dd -= dd.min()
            dd /= dd.max()
            c1, c2 = np.tile(n1["color"], dd.shape), np.tile(n2["color"], dd.shape)
            c_grad = (1 - dd) * c1 + dd * c2
            rr, cc = rr.clip(0, sprite.shape[0] - 1), cc.clip(0, sprite.shape[1] - 1)
            sprite[rr, cc] = c_grad.astype(np.uint8)
        sprite = Image.fromarray(sprite, "RGBA")
        draw = ImageDraw.Draw(sprite)
        for node in lattice.nodes:
            draw.ellipse(
                xy=[
                    tuple(lattice.nodes[node]["pos"] - thickness // 2),
                    tuple(lattice.nodes[node]["pos"] + thickness // 2),
                ],
                fill=tuple(lattice.nodes[node]["color"]),
            )
        return sprite


object_types = [Arm, Hand, Lattice, Polygon, Pie, Scissors]
