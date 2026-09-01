"""Pedagogical sequential CPU 3D Gaussian Splatting renderer.

Usage:
    python 3dgs_renderer_v1.py point_cloud.ply render.png

The input must use the conventional 3DGS vertex properties:
x, y, z, opacity, scale_0..2, rot_0..3, and f_dc_0..2.
This reference renderer is intentionally simple: every pixel tests every
projected splat after a global centre-depth sort. It is correct enough for
learning, but not suitable for large scenes. Unit 4 explains tiled binning.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# `shared/` sits beside this file in the assignment repo, and one level up in the course repo.
for _candidate in (_here / "shared", _here.parent / "shared"):
    if _candidate.is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from camera import Camera
from gaussian_set import GaussianSet
from projected_gaussians import ProjectedGaussians
from splat_math import ALPHA_CUTOFF, SUPPORT_RADIUS_SQUARED, quaternion_to_matrix
# A faint splat reaches ALPHA_CUTOFF sooner than a strong one, so it can be truncated sooner than
# the full 3-sigma disc. `beta` scales how quickly that happens. Note the Warp trainers use
# `compact_box.beta = 0.5`; these renderers keep 1.0, which is what v2 and v3 have always applied.
COMPACT_BOX_BETA = 1.0


def compact_support(opacities: np.ndarray) -> np.ndarray:
    """Per-splat squared-Mahalanobis cutoff.

    All three renderers must use this one definition. v1 previously tested against a fixed
    SUPPORT_RADIUS_SQUARED while v2 and v3 tested against this compact support, so the "ports"
    truncated faint splats earlier than the reference they were meant to reproduce.
    """
    opacities = np.asarray(opacities, dtype=np.float32)
    supports = np.zeros(len(opacities), dtype=np.float32)
    visible = opacities > ALPHA_CUTOFF
    supports[visible] = np.minimum(
        SUPPORT_RADIUS_SQUARED,
        COMPACT_BOX_BETA * 2.0 * np.log(opacities[visible] / ALPHA_CUTOFF),
    )
    return supports


def project_gaussians(
    splats: GaussianSet,
    camera: Camera,
    near: float = 0.01,
    filter_variance: float = 0.3,
    qmax: float = SUPPORT_RADIUS_SQUARED,
) -> ProjectedGaussians:
    """Build compact screen-space records and sort them front to back."""
    rotation_world = quaternion_to_matrix(splats.rotations)
    scale_squared = splats.scales * splats.scales
    covariance_world = np.einsum(
        "nij,nj,nkj->nik", rotation_world, scale_squared, rotation_world
    )

    world_h = np.concatenate(
        (splats.means, np.ones((len(splats.means), 1), dtype=np.float32)), axis=1
    )
    camera_h = world_h @ camera.world_to_camera.T
    mean_camera = camera_h[:, :3]
    depth = mean_camera[:, 2]

    W = camera.world_to_camera[:3, :3]
    covariance_camera = W @ covariance_world @ W.T

    x, y, z = mean_camera.T
    centres = np.stack(
        (camera.fx * x / z + camera.cx, camera.fy * y / z + camera.cy), axis=1
    )

    jacobian = np.zeros((len(splats.means), 2, 3), dtype=np.float32)
    jacobian[:, 0, 0] = camera.fx / z
    jacobian[:, 0, 2] = -camera.fx * x / (z * z)
    jacobian[:, 1, 1] = camera.fy / z
    jacobian[:, 1, 2] = -camera.fy * y / (z * z)
    covariance_screen = jacobian @ covariance_camera @ np.swapaxes(jacobian, 1, 2)
    covariance_screen += filter_variance * np.eye(2, dtype=np.float32)

    eigenvalues = np.linalg.eigvalsh(covariance_screen)
    radii = np.sqrt(qmax * np.maximum(eigenvalues[:, 1], 0.0))
    visible = (
        (depth > near)
        & (centres[:, 0] + radii >= 0.0)
        & (centres[:, 0] - radii < camera.width)
        & (centres[:, 1] + radii >= 0.0)
        & (centres[:, 1] - radii < camera.height)
    )

    indices = np.flatnonzero(visible)
    if len(indices) == 0:
        return ProjectedGaussians(
            np.empty((0, 2), np.float32),
            np.empty((0, 3), np.float32),
            np.empty((0,), np.float32),
            np.empty((0, 3), np.float32),
            np.empty((0,), np.float32),
        )

    # The global sort is only pedagogical. A scalable renderer sorts per tile.
    # Sort the visible-splat indices once, up front, so everything below is built in a single
    # index space. Deriving `conics` before the sort and then indexing it by `order` while the
    # other fields are indexed by the reassigned `indices` is equivalent, but leaves two
    # permutations of the same data four lines apart and silently desynchronises if either moves.
    indices = indices[np.argsort(depth[indices])]

    inverse = np.linalg.inv(covariance_screen[indices])
    conics = np.stack((inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1]), axis=1)
    return ProjectedGaussians(
        centres[indices].astype(np.float32),
        conics.astype(np.float32),
        depth[indices].astype(np.float32),
        splats.colors[indices].astype(np.float32),
        splats.opacities[indices].astype(np.float32),
    )


class CpuRenderer:
    """A sequential reference renderer: rows, pixels, then sorted splats."""

    def __init__(
        self,
        camera: Camera,
        near: float = 0.01,
        filter_variance: float = 0.3,
        qmax: float = SUPPORT_RADIUS_SQUARED,
    ):
        self.camera = camera
        self.near = near
        self.filter_variance = filter_variance
        self.qmax = qmax

    def render(
        self,
        splats: GaussianSet,
        background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        projected = project_gaussians(
            splats,
            self.camera,
            near=self.near,
            filter_variance=self.filter_variance,
            qmax=self.qmax,
        )
        image = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.float32)
        background_color = np.asarray(background, dtype=np.float32)
        supports = compact_support(projected.opacities)

        # Deliberately sequential: one row, one pixel, and one splat at a time.
        for py in range(self.camera.height):
            y = py + 0.5
            for px in range(self.camera.width):
                x = px + 0.5
                colour = np.zeros(3, dtype=np.float32)
                transmittance = 1.0

                for i in range(len(projected.centres)):
                    cx, cy = projected.centres[i]
                    du = x - cx
                    dv = y - cy
                    a, b, c = projected.conics[i]

                    q = a * du * du + 2.0 * b * du * dv + c * dv * dv

                    if supports[i] == 0.0 or q > supports[i]:
                        continue

                    alpha = min(0.99, projected.opacities[i] * np.exp(-0.5 * q))

                    if alpha < ALPHA_CUTOFF:
                        continue

                    colour += transmittance * alpha * projected.colors[i]
                    transmittance *= 1.0 - alpha

                    if transmittance < 1.0e-4:
                        break

                colour += transmittance * background_color
                image[py, px] = colour


                # TODO: The RHS is a placeholder
                # image[py, px] = np.zeros(3, dtype=np.float32)

        return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="3DGS point_cloud.ply file")
    parser.add_argument("output", help="output PNG file")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--focal-length", type=float, default=700.0)
    parser.add_argument("--camera-position", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--look-at", nargs=3, type=float, default=(0.0, 0.0, 1.0))
    parser.add_argument("--up", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    args = parser.parse_args()

    camera = Camera.from_look_at(
        args.width, args.height, args.focal_length,
        args.camera_position, args.look_at, args.up,
    )
    splats = GaussianSet.from_ply(args.ply)
    image = CpuRenderer(camera).render(splats)

    image_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_u8).save(args.output)
    # Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0)).save(args.output)


if __name__ == "__main__":
    main()
