"""Geometric objects for raytracing."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from terminalray.math3d import Vec3, Ray, Hit, dot, normalize


@dataclass
class Sphere:
    """A sphere defined by center and radius."""
    center: Vec3
    radius: float
    material: object

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """
        Ray-sphere intersection using quadratic formula.
        Returns Hit if intersection found, None otherwise.
        """
        oc = ray.origin - self.center
        a = dot(ray.direction, ray.direction)
        b = 2.0 * dot(oc, ray.direction)
        c = dot(oc, oc) - self.radius * self.radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None

        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2 * a)
        t2 = (-b + sqrt_d) / (2 * a)

        # Find the nearest positive t
        t = t1 if t1 > 1e-6 else t2
        if t < 1e-6:
            return None

        point = ray.at(t)
        normal = normalize(point - self.center)

        return Hit(t=t, point=point, normal=normal, material=self.material)


@dataclass
class Plane:
    """An infinite plane defined by a point and normal."""
    point: Vec3
    normal: Vec3
    material: object

    def __post_init__(self):
        self.normal = normalize(self.normal)

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """
        Ray-plane intersection.
        Returns Hit if intersection found, None otherwise.
        """
        denom = dot(ray.direction, self.normal)
        if abs(denom) < 1e-6:
            return None  # Ray is parallel to plane

        t = dot(self.point - ray.origin, self.normal) / denom
        if t < 1e-6:
            return None  # Intersection is behind ray origin

        point = ray.at(t)

        # Return normal facing the ray
        normal = self.normal if denom < 0 else -self.normal

        return Hit(t=t, point=point, normal=normal, material=self.material)
