"""Gaussian-first tiled Warp 3DGS renderer.

Usage:
    python 3dgs_renderer_v3.py point_cloud.ply render.png --device cpu

This version uses Gaussian-first traversal to build tile lists: each projected Gaussian emits a
record for every screen-space tile touched by its bounding box. The raster kernel then launches one
work item per pixel and composites only the sorted Gaussian list for that pixel's tile.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import warp as wp

_reference = importlib.import_module("3dgs_renderer_v1")
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# `shared/` sits beside this file in the assignment repo, and one level up in the course repo.
for _candidate in (_here / "shared", _here.parent / "shared"):
    if _candidate.is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from tile_builder import GaussianFirstTileBuilder

Camera = _reference.Camera
GaussianSet = _reference.GaussianSet
project_gaussians = _reference.project_gaussians
SUPPORT_RADIUS_SQUARED = _reference.SUPPORT_RADIUS_SQUARED
compact_support = _reference.compact_support
ALPHA_CUTOFF = _reference.ALPHA_CUTOFF
TRANSMITTANCE_CUTOFF = 1.0e-4


@wp.kernel
def rasterize_tiles(
    centres: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec3),
    colours: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=wp.float32),
    supports: wp.array(dtype=wp.float32),
    tile_offsets: wp.array(dtype=wp.int32),
    packed_pairs: wp.array(dtype=wp.uint64),
    width: int,
    tile_size: int,
    tiles_x: int,
    background: wp.vec3,
    image: wp.array(dtype=wp.vec3),
):
    pixel = wp.tid()
    px_i = pixel % width
    py_i = pixel // width

    tile_x = px_i // tile_size
    tile_y = py_i // tile_size
    tile = tile_y * tiles_x + tile_x

    px = float(px_i) + 0.5
    py = float(py_i) + 0.5
    colour = wp.vec3(0.0, 0.0, 0.0)
    transmittance = 1.0


    # TODO: Compute the RGB value at image[pixel].
    # Composite only this pixel's tile list, `tile_offsets[tile] .. tile_offsets[tile + 1]`,
    # which the builder has already sorted near to far. Finish with the background
    # weighted by the remaining transmittance, matching 3dgs_renderer_v1.

    start = tile_offsets[tile]
    end = tile_offsets[tile + 1]
    for record_index in range(start, end):
        packed = packed_pairs[record_index]
        splat = int(packed & wp.uint64(0xFFFFFFFF)) # bitwise AND to get the lower 32 bits, which is the splat index

        centre = centres[splat]
        conic = conics[splat]
        du = px - centre[0]
        dv = py - centre[1]
        q = conic[0] * du * du + 2.0 * conic[1] * du * dv + conic[2] * dv * dv

        if q > supports[splat]:
            continue

        alpha = wp.min(0.99, opacities[splat] * wp.exp(-0.5 * q))
        if alpha < ALPHA_CUTOFF:
            continue

        colour += transmittance * alpha * colours[splat]
        transmittance *= 1.0 - alpha

        if transmittance < TRANSMITTANCE_CUTOFF:
            break

    image[pixel] = colour + transmittance * background


class GaussianFirstWarpRenderer:
    """Persistent Warp storage for Gaussian-first tile assignment and tiled rasterization."""

    def __init__(
        self,
        width: int,
        height: int,
        maximum_splats: int,
        device: str,
        tile_size: int = 16,
        tile_pair_capacity: int | None = None,
    ):
        self.width = width
        self.height = height
        self.maximum_splats = maximum_splats
        self.tile_size = tile_size
        self.tiles_x = (width + tile_size - 1) // tile_size
        self.tiles_y = (height + tile_size - 1) // tile_size
        self.tile_count = self.tiles_x * self.tiles_y
        self.tile_pair_capacity = (
            int(tile_pair_capacity)
            if tile_pair_capacity is not None
            else max(1, maximum_splats * 32)
        )
        self.device = wp.get_device(device)

        self.centres = wp.zeros(maximum_splats, dtype=wp.vec2, device=self.device)
        self.conics = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
        self.colours = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
        self.opacities = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.supports = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.depths = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.group_ids = wp.zeros(maximum_splats, dtype=wp.int32, device=self.device)
        self.splat_ids = wp.array(
            np.arange(maximum_splats, dtype=np.uint32), dtype=wp.uint32, device=self.device
        )
        self.image = wp.zeros(width * height, dtype=wp.vec3, device=self.device)
        self.builder = GaussianFirstTileBuilder(
            maximum_splats,
            self.tile_count,
            width,
            height,
            tile_size,
            self.tile_pair_capacity,
            self.device,
        )

    def render(self, splats: GaussianSet, camera: Camera,
               background: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        projected = project_gaussians(splats, camera)
        count = len(projected.opacities)

        if count > self.maximum_splats:
            raise ValueError(
                f"Renderer capacity {self.maximum_splats:,} is below {count:,} visible splats."
            )

        self.centres.assign(wp.array(projected.centres, dtype=wp.vec2, device=self.device))
        self.conics.assign(wp.array(projected.conics, dtype=wp.vec3, device=self.device))
        self.colours.assign(wp.array(projected.colors, dtype=wp.vec3, device=self.device))
        self.opacities.assign(wp.array(projected.opacities, dtype=wp.float32, device=self.device))
        supports = compact_support(projected.opacities)
        self.supports.assign(wp.array(supports, dtype=wp.float32, device=self.device))
        self.depths.assign(wp.array(projected.depths, dtype=wp.float32, device=self.device))

        offsets, packed_pairs, _ = self.builder.build(
            self.centres,
            self.conics,
            self.supports,
            self.depths,
            self.group_ids,
            self.splat_ids,
            count,
            self.tile_count,
        )

        wp.launch(
            rasterize_tiles,
            dim=self.width * self.height,
            inputs=[
                self.centres,
                self.conics,
                self.colours,
                self.opacities,
                self.supports,
                offsets,
                packed_pairs,
                self.width,
                self.tile_size,
                self.tiles_x,
                wp.vec3(*background),
                self.image,
            ],
            device=self.device,
        )

        return self.image.numpy().reshape(self.height, self.width, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("output")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--focal-length", type=float, default=350.0)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--tile-pair-capacity", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="Warp device, such as cpu or cuda:0")
    parser.add_argument("--camera-position", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--look-at", nargs=3, type=float, default=(0.0, 0.0, 1.0))
    parser.add_argument("--up", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    args = parser.parse_args()

    wp.init()
    camera = Camera.from_look_at(
        args.width,
        args.height,
        args.focal_length,
        args.camera_position,
        args.look_at,
        args.up,
    )
    splats = GaussianSet.from_ply(args.ply)
    renderer = GaussianFirstWarpRenderer(
        args.width,
        args.height,
        len(splats.means),
        args.device,
        args.tile_size,
        args.tile_pair_capacity,
    )
    image = renderer.render(splats, camera)
    Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0)).save(args.output)


if __name__ == "__main__":
    main()
