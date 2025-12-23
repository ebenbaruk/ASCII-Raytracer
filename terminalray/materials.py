"""Material definitions for raytracing."""

from dataclasses import dataclass
import numpy as np
from terminalray.math3d import Vec3, vec3


@dataclass
class DiffuseMaterial:
    """A matte (Lambertian) material with Phong specular."""
    color: Vec3
    shininess: float = 32.0
    specular: float = 0.5

    @property
    def reflectivity(self) -> float:
        return 0.0


@dataclass
class MirrorMaterial:
    """A reflective mirror-like material."""
    reflectivity: float = 0.9
    color: Vec3 = None
    shininess: float = 128.0
    specular: float = 1.0

    def __post_init__(self):
        if self.color is None:
            self.color = vec3(1.0, 1.0, 1.0)


@dataclass
class CheckerMaterial:
    """A checkerboard pattern material for floors."""
    color1: Vec3
    color2: Vec3
    scale: float = 1.0
    shininess: float = 16.0
    specular: float = 0.3

    @property
    def reflectivity(self) -> float:
        return 0.0

    def get_color(self, x: float, z: float) -> Vec3:
        """Get the color at a given (x, z) position on the floor."""
        ix = int(np.floor(x / self.scale))
        iz = int(np.floor(z / self.scale))
        if (ix + iz) % 2 == 0:
            return self.color1
        return self.color2
