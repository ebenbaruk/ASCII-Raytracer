"""Vector math utilities and core data structures for raytracing."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]


def vec3(x: float, y: float, z: float) -> Vec3:
    """Create a 3D vector."""
    return np.array([x, y, z], dtype=np.float64)


def length(v: Vec3) -> float:
    """Return the length (magnitude) of a vector."""
    return np.sqrt(np.dot(v, v))


def normalize(v: Vec3) -> Vec3:
    """Return the unit vector in the same direction."""
    l = length(v)
    if l < 1e-10:
        return vec3(0, 0, 0)
    return v / l


def dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two vectors."""
    return np.dot(a, b)


def cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two vectors."""
    return np.cross(a, b)


def reflect(v: Vec3, n: Vec3) -> Vec3:
    """Reflect vector v around normal n (n should be normalized)."""
    return v - 2 * dot(v, n) * n


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, x))


def clamp_vec(v: Vec3, lo: float = 0.0, hi: float = 1.0) -> Vec3:
    """Clamp each component of a vector between lo and hi."""
    return np.clip(v, lo, hi)


@dataclass
class Ray:
    """A ray with origin and direction."""
    origin: Vec3
    direction: Vec3

    def at(self, t: float) -> Vec3:
        """Get the point along the ray at parameter t."""
        return self.origin + t * self.direction


@dataclass
class Hit:
    """Information about a ray-object intersection."""
    t: float           # Distance along ray
    point: Vec3        # Hit point in world space
    normal: Vec3       # Surface normal at hit point (normalized, outward)
    material: object   # Material at hit point
