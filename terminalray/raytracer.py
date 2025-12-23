"""Core raytracing engine with shading, shadows, and reflections."""

import numpy as np
from typing import Optional

from terminalray.math3d import (
    Vec3, vec3, Ray, Hit,
    normalize, dot, reflect, clamp, clamp_vec, length
)
from terminalray.scene import Scene
from terminalray.materials import CheckerMaterial, MirrorMaterial


class Camera:
    """A perspective camera for ray generation."""

    def __init__(
        self,
        position: Vec3,
        look_at: Vec3,
        up: Vec3 = None,
        fov: float = 60.0,
        aspect_ratio: float = 2.0  # Terminal chars are ~2:1 tall
    ):
        if up is None:
            up = vec3(0, 1, 0)

        self.position = position
        self.look_at = look_at
        self.fov = fov
        self.aspect_ratio = aspect_ratio

        # Build camera coordinate frame
        self.forward = normalize(look_at - position)
        self.right = normalize(np.cross(self.forward, up))
        self.up = np.cross(self.right, self.forward)

        # Compute view plane dimensions
        self.half_height = np.tan(np.radians(fov / 2))
        self.half_width = self.half_height * aspect_ratio

    def get_ray(self, u: float, v: float) -> Ray:
        """
        Generate a ray for normalized screen coordinates.
        u, v are in range [0, 1], where (0,0) is top-left.
        """
        # Convert to [-1, 1] range, with y inverted
        x = (2 * u - 1) * self.half_width
        y = (1 - 2 * v) * self.half_height

        direction = normalize(
            self.forward + x * self.right + y * self.up
        )
        return Ray(origin=self.position.copy(), direction=direction)


class Raytracer:
    """The main raytracing renderer."""

    def __init__(self, scene: Scene, max_depth: int = 1):
        self.scene = scene
        self.max_depth = max_depth

    def trace(self, ray: Ray, depth: int = 0) -> Vec3:
        """
        Trace a ray through the scene and return the color.
        """
        hit = self.scene.intersect(ray)

        if hit is None:
            return self.scene.background.copy()

        # Get material color (handle checkerboard)
        material = hit.material
        if isinstance(material, CheckerMaterial):
            base_color = material.get_color(hit.point[0], hit.point[2])
        else:
            base_color = material.color.copy()

        # Start with ambient
        color = self.scene.ambient * base_color

        # Add contribution from each light
        for light in self.scene.lights:
            color += self._shade(hit, light, ray, base_color)

        # Add reflection if material is reflective and we haven't hit max depth
        reflectivity = getattr(material, 'reflectivity', 0.0)
        if reflectivity > 0 and depth < self.max_depth:
            reflected_dir = reflect(ray.direction, hit.normal)
            reflected_ray = Ray(
                origin=hit.point + hit.normal * 1e-4,  # Bias to avoid self-intersection
                direction=reflected_dir
            )
            reflected_color = self.trace(reflected_ray, depth + 1)
            color = (1 - reflectivity) * color + reflectivity * reflected_color

        return clamp_vec(color)

    def _shade(self, hit: Hit, light, ray: Ray, base_color: Vec3) -> Vec3:
        """Compute shading for a single light (Blinn-Phong model)."""
        # Direction to light
        to_light = light.position - hit.point
        light_dist = length(to_light)
        light_dir = to_light / light_dist

        # Shadow test
        shadow_ray = Ray(
            origin=hit.point + hit.normal * 1e-4,
            direction=light_dir
        )
        shadow_hit = self.scene.intersect(shadow_ray)
        if shadow_hit is not None and shadow_hit.t < light_dist:
            # In shadow - only return a tiny bit of ambient
            return vec3(0, 0, 0)

        # Diffuse (Lambert)
        n_dot_l = max(0.0, dot(hit.normal, light_dir))
        diffuse = n_dot_l * base_color * light.intensity

        # Specular (Blinn-Phong)
        view_dir = normalize(-ray.direction)
        half_vec = normalize(light_dir + view_dir)
        n_dot_h = max(0.0, dot(hit.normal, half_vec))

        material = hit.material
        shininess = getattr(material, 'shininess', 32.0)
        specular_strength = getattr(material, 'specular', 0.5)
        specular = specular_strength * (n_dot_h ** shininess) * light.color * light.intensity

        return diffuse + specular

    def render_frame(
        self,
        camera: Camera,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Render a full frame and return an array of colors.
        Returns shape (height, width, 3) with RGB values in [0, 1].
        """
        frame = np.zeros((height, width, 3), dtype=np.float64)

        for y in range(height):
            v = y / height
            for x in range(width):
                u = x / width
                ray = camera.get_ray(u, v)
                frame[y, x] = self.trace(ray)

        return frame
