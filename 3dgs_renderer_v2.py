"""Warp port of the Unit 5 parallel 3DGS raster stage.

Usage:
    python 3dgs_renderer_v2.py point_cloud.ply render.png --device cpu

Projection and global near-to-far ordering deliberately reuse the sequential reference. Warp
owns persistent screen-space arrays and launches one parallel work item per output pixel.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
from PIL import Image
import warp as wp

_reference = importlib.import_module("3dgs_renderer_v1")
Camera = _reference.Camera
GaussianSet = _reference.GaussianSet
project_gaussians = _reference.project_gaussians
SUPPORT_RADIUS_SQUARED = _reference.SUPPORT_RADIUS_SQUARED
compact_support = _reference.compact_support
ALPHA_CUTOFF = _reference.ALPHA_CUTOFF
TRANSMITTANCE_CUTOFF = 1.0e-4


@wp.kernel
def rasterize(
    centres: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec3),
    colours: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=wp.float32),
    supports: wp.array(dtype=wp.float32),
    count: int,
    width: int,
    background: wp.vec3,
    image: wp.array(dtype=wp.vec3),
):
    pixel = wp.tid()
    px = float(pixel % width) + 0.5
    py = float(pixel // width) + 0.5
    # TODO: Calculate the RGB at pixel (px, py)
    # One work item per pixel: walk the globally depth-sorted splats, accumulate
    # front to back, and composite the background with the leftover transmittance.
    # This must reproduce 3dgs_renderer_v1 exactly.

    # TODO: The RHS is a placeholder
    image[pixel] = wp.vec3(0.0, 0.0, 0.0)


class WarpRenderer:
    """Persistent Warp storage for the screen-space records and rendered pixels."""

    def __init__(self, width: int, height: int, maximum_splats: int, device: str):
        self.width, self.height, self.maximum_splats = width, height, maximum_splats
        self.device = wp.get_device(device)
        self.centres = wp.zeros(maximum_splats, dtype=wp.vec2, device=self.device)
        self.conics = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
        self.colours = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
        self.opacities = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.supports = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.image = wp.zeros(width * height, dtype=wp.vec3, device=self.device)

    def render(self, splats: GaussianSet, camera: Camera,
               background: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        projected = project_gaussians(splats, camera)
        count = len(projected.opacities)
        if count > self.maximum_splats:
            raise ValueError(f"Renderer capacity {self.maximum_splats:,} is below {count:,} visible splats.")
        self.centres.assign(wp.array(projected.centres, dtype=wp.vec2, device=self.device))
        self.conics.assign(wp.array(projected.conics, dtype=wp.vec3, device=self.device))
        self.colours.assign(wp.array(projected.colors, dtype=wp.vec3, device=self.device))
        self.opacities.assign(wp.array(projected.opacities, dtype=wp.float32, device=self.device))
        supports = compact_support(projected.opacities)
        self.supports.assign(wp.array(supports, dtype=wp.float32, device=self.device))
        wp.launch(rasterize, dim=self.width * self.height,
                  inputs=[self.centres, self.conics, self.colours, self.opacities, self.supports,
                          count, self.width, wp.vec3(*background), self.image],
                  device=self.device)
        return self.image.numpy().reshape(self.height, self.width, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("output")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--focal-length", type=float, default=350.0)
    parser.add_argument("--device", default="cpu", help="Warp device, such as cpu or cuda:0")
    parser.add_argument("--camera-position", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--look-at", nargs=3, type=float, default=(0.0, 0.0, 1.0))
    parser.add_argument("--up", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    args = parser.parse_args()

    wp.init()
    camera = Camera.from_look_at(
        args.width, args.height, args.focal_length,
        args.camera_position, args.look_at, args.up,
    )
    splats = GaussianSet.from_ply(args.ply)
    renderer = WarpRenderer(args.width, args.height, len(splats.means), args.device)
    image = renderer.render(splats, camera)
    # Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0)).save(args.output)
    
    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_u8).save(args.output)

if __name__ == "__main__":
    main()
