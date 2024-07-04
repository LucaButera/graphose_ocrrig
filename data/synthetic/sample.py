from typing import Tuple

import networkx as nx
import numpy as np
from numpy.random import Generator
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist

from data.synthetic.objects import object_types
from data.synthetic.primitives import BG_COLOR
from utils.data import graphose_rng


class SyntheticObjectsSample:
    def __init__(
        self,
        side_len: int = 1000,
        max_objects: int = 4,
        max_trials: int = 1000,
        rng: Generator = graphose_rng,
    ):
        self.side_len = side_len
        n_objects = rng.integers(1, max_objects, endpoint=True).item()
        max_radius = (
            self.side_len / 2 / (n_objects if (n_objects < 2 or n_objects > 4) else 2)
        )
        radius_variance = 0.75 if n_objects == 1 else 0.9
        quadrants = [
            (0, 0),
            (self.side_len / 2, 0),
            (0, self.side_len / 2),
            (self.side_len / 2, self.side_len / 2),
        ]
        rng.shuffle(quadrants)
        objects, positions, radii = [], [], []
        for i in range(n_objects):
            placed = False
            trials = 0
            curr_max_radius = max_radius * i ** (-0.1) if i > 0 else max_radius
            while not placed and trials < max_trials:
                radius = int(
                    rng.uniform(radius_variance * curr_max_radius, curr_max_radius)
                )
                if n_objects < 2 or n_objects > 4:
                    position = np.floor(
                        rng.uniform(radius, self.side_len - radius + 1, 2)
                    )
                else:
                    position = (
                        np.floor(rng.uniform(radius, self.side_len / 2 - radius + 1, 2))
                        + quadrants[i]
                    )
                trials += 1
                if i == 0 or not np.any(
                    cdist(position[None], np.array(positions))
                    <= (radius + np.array(radii))
                ):
                    obj_type = rng.choice(object_types)
                    obj = obj_type(position=position, radius=radius, rng=rng)
                    objects.append(obj)
                    positions.append(position)
                    radii.append(radius)
                    placed = True
        self.objects = objects

    def get_graph(self) -> nx.Graph:
        whole_graph = nx.Graph()
        for obj in self.objects:
            g: nx.Graph = obj.graph.copy()
            nx.set_node_attributes(
                g,
                {
                    k: (
                        g.nodes[k]["pos"]
                        + np.array(obj.position)
                        - np.array((obj.radius, obj.radius))
                    )
                    / np.array((self.side_len, self.side_len))
                    for k in g
                },
                "pos",
            )
            whole_graph = nx.disjoint_union(whole_graph, g)
        return whole_graph

    def get_image(self) -> Image.Image:
        bg = Image.new("RGBA", (self.side_len, self.side_len), BG_COLOR)
        for obj in self.objects:
            sprite = obj.sprite
            pos = (
                int(obj.position[0] - sprite.width / 2),
                int(obj.position[1] - sprite.height / 2),
            )
            bg.alpha_composite(sprite, pos)
        return bg

    def get_sample(self) -> Tuple[nx.Graph, Image.Image]:
        return self.get_graph(), self.get_image()

    def display(self, show_graph: bool = True):
        graph, img = self.get_sample()
        if show_graph:
            draw = ImageDraw.Draw(img)
            scale = np.array((self.side_len, self.side_len))
            for e in graph.edges:
                p1, p2 = graph.nodes[e[0]]["pos"], graph.nodes[e[1]]["pos"]
                draw.line(
                    xy=[tuple(p1 * scale), tuple(p2 * scale)],
                    fill=(255, 255, 255, 255),
                    width=2,
                )
            for n in graph:
                draw.ellipse(
                    xy=[
                        tuple(graph.nodes[n]["pos"] * scale - 2),
                        tuple(graph.nodes[n]["pos"] * scale + 2),
                    ],
                    fill=(255, 255, 255, 255),
                    outline=(0, 0, 0, 255),
                    width=2,
                )
        img.show()
