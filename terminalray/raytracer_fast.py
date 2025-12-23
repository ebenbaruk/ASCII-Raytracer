"""Numba-optimized raytracing engine for maximum performance."""

import numpy as np
from numba import njit, prange
from typing import Tuple

# Constants for material types
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_CHECKER = 2


@njit(cache=True)
def normalize(v):
    """Normalize a vector."""
    l = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if l < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    return v / l


@njit(cache=True)
def dot(a, b):
    """Dot product."""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@njit(cache=True)
def reflect(v, n):
    """Reflect vector v around normal n."""
    return v - 2.0 * dot(v, n) * n


@njit(cache=True)
def intersect_sphere(ray_o, ray_d, center, radius):
    """
    Ray-sphere intersection.
    Returns (hit, t, normal) where hit is boolean.
    """
    oc = ray_o - center
    a = dot(ray_d, ray_d)
    b = 2.0 * dot(oc, ray_d)
    c = dot(oc, oc) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False, 0.0, np.array([0.0, 0.0, 0.0])

    sqrt_d = np.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)

    t = t1 if t1 > 1e-4 else t2
    if t < 1e-4:
        return False, 0.0, np.array([0.0, 0.0, 0.0])

    point = ray_o + t * ray_d
    normal = normalize(point - center)
    return True, t, normal


@njit(cache=True)
def intersect_plane(ray_o, ray_d, plane_point, plane_normal):
    """
    Ray-plane intersection.
    Returns (hit, t, normal).
    """
    denom = dot(ray_d, plane_normal)
    if abs(denom) < 1e-6:
        return False, 0.0, np.array([0.0, 0.0, 0.0])

    t = dot(plane_point - ray_o, plane_normal) / denom
    if t < 1e-4:
        return False, 0.0, np.array([0.0, 0.0, 0.0])

    normal = plane_normal if denom < 0 else -plane_normal
    return True, t, normal


@njit(cache=True)
def get_checker_color(x, z, scale, color1, color2):
    """Get checkerboard color at position."""
    ix = int(np.floor(x / scale))
    iz = int(np.floor(z / scale))
    if (ix + iz) % 2 == 0:
        return color1
    return color2


@njit(cache=True)
def trace_scene(
    ray_o, ray_d,
    # Sphere 1 (mirror)
    s1_center, s1_radius, s1_reflectivity, s1_color,
    # Sphere 2 (diffuse)
    s2_center, s2_radius, s2_color,
    # Plane (checker)
    plane_point, plane_normal, checker_scale, checker_c1, checker_c2,
    # Light
    light_pos, light_intensity,
    # Settings
    ambient, background, max_depth,
    # Effects
    fog_density, fog_color
):
    """
    Trace a ray through the scene.
    Returns RGB color as numpy array.
    """
    color = np.array([0.0, 0.0, 0.0])
    attenuation = np.array([1.0, 1.0, 1.0])

    current_ray_o = ray_o.copy()
    current_ray_d = ray_d.copy()

    for depth in range(max_depth + 1):
        # Find closest intersection
        closest_t = 1e20
        hit_type = -1  # 0=sphere1, 1=sphere2, 2=plane
        hit_normal = np.array([0.0, 0.0, 0.0])
        hit_point = np.array([0.0, 0.0, 0.0])

        # Test sphere 1 (mirror)
        hit, t, normal = intersect_sphere(current_ray_o, current_ray_d, s1_center, s1_radius)
        if hit and t < closest_t:
            closest_t = t
            hit_type = 0
            hit_normal = normal
            hit_point = current_ray_o + t * current_ray_d

        # Test sphere 2 (diffuse)
        hit, t, normal = intersect_sphere(current_ray_o, current_ray_d, s2_center, s2_radius)
        if hit and t < closest_t:
            closest_t = t
            hit_type = 1
            hit_normal = normal
            hit_point = current_ray_o + t * current_ray_d

        # Test plane
        hit, t, normal = intersect_plane(current_ray_o, current_ray_d, plane_point, plane_normal)
        if hit and t < closest_t:
            closest_t = t
            hit_type = 2
            hit_normal = normal
            hit_point = current_ray_o + t * current_ray_d

        # No hit - return background
        if hit_type < 0:
            color = color + attenuation * background
            break

        # Get material color
        if hit_type == 0:  # Mirror sphere
            base_color = s1_color.copy()
            reflectivity = s1_reflectivity
            shininess = 128.0
            specular = 1.0
        elif hit_type == 1:  # Diffuse sphere
            base_color = s2_color.copy()
            reflectivity = 0.0
            shininess = 32.0
            specular = 0.4
        else:  # Checker plane
            base_color = get_checker_color(hit_point[0], hit_point[2], checker_scale, checker_c1, checker_c2)
            reflectivity = 0.0
            shininess = 16.0
            specular = 0.2

        # Lighting calculation
        to_light = light_pos - hit_point
        light_dist = np.sqrt(dot(to_light, to_light))
        light_dir = to_light / light_dist

        # Shadow test
        shadow_origin = hit_point + hit_normal * 1e-3
        in_shadow = False

        # Check shadow against sphere 1
        if hit_type != 0:
            sh, st, _ = intersect_sphere(shadow_origin, light_dir, s1_center, s1_radius)
            if sh and st < light_dist:
                in_shadow = True

        # Check shadow against sphere 2
        if not in_shadow and hit_type != 1:
            sh, st, _ = intersect_sphere(shadow_origin, light_dir, s2_center, s2_radius)
            if sh and st < light_dist:
                in_shadow = True

        # Shading
        surface_color = ambient * base_color

        if not in_shadow:
            # Diffuse
            n_dot_l = max(0.0, dot(hit_normal, light_dir))
            surface_color = surface_color + n_dot_l * base_color * light_intensity

            # Specular (Blinn-Phong)
            view_dir = normalize(-current_ray_d)
            half_vec = normalize(light_dir + view_dir)
            n_dot_h = max(0.0, dot(hit_normal, half_vec))
            spec = specular * (n_dot_h ** shininess) * light_intensity
            surface_color = surface_color + np.array([spec, spec, spec])

        # Apply fog
        if fog_density > 0:
            fog_factor = np.exp(-fog_density * closest_t)
            surface_color = fog_factor * surface_color + (1.0 - fog_factor) * fog_color

        # Add to accumulated color
        color = color + attenuation * (1.0 - reflectivity) * surface_color

        # Handle reflection
        if reflectivity > 0 and depth < max_depth:
            attenuation = attenuation * reflectivity
            current_ray_o = hit_point + hit_normal * 1e-3
            current_ray_d = reflect(current_ray_d, hit_normal)
        else:
            break

    # Clamp
    color[0] = min(1.0, max(0.0, color[0]))
    color[1] = min(1.0, max(0.0, color[1]))
    color[2] = min(1.0, max(0.0, color[2]))

    return color


@njit(parallel=True, cache=True)
def render_frame_fast(
    width, height,
    # Camera
    cam_pos, cam_forward, cam_right, cam_up, half_width, half_height,
    # Sphere 1 (mirror)
    s1_center, s1_radius, s1_reflectivity, s1_color,
    # Sphere 2 (diffuse)
    s2_center, s2_radius, s2_color,
    # Plane
    plane_point, plane_normal, checker_scale, checker_c1, checker_c2,
    # Light
    light_pos, light_intensity,
    # Settings
    ambient, background, max_depth,
    # Effects
    fog_density, fog_color,
    # DOF settings (0 = disabled)
    dof_aperture, dof_focal_dist, dof_samples,
    # Soft shadow settings (1 = hard shadows)
    shadow_samples, light_radius
):
    """
    Render a full frame with parallel processing.
    Returns (height, width, 3) array.
    """
    frame = np.zeros((height, width, 3))

    for y in prange(height):
        for x in range(width):
            u = x / width
            v = y / height

            # Convert to [-1, 1] range
            px = (2 * u - 1) * half_width
            py = (1 - 2 * v) * half_height

            direction = normalize(cam_forward + px * cam_right + py * cam_up)

            if dof_aperture > 0 and dof_samples > 1:
                # Depth of field - average multiple samples
                color = np.array([0.0, 0.0, 0.0])
                focal_point = cam_pos + direction * dof_focal_dist

                for _ in range(dof_samples):
                    # Random point on aperture disk
                    angle = np.random.random() * 2 * np.pi
                    r = np.sqrt(np.random.random()) * dof_aperture
                    offset = r * np.cos(angle) * cam_right + r * np.sin(angle) * cam_up

                    dof_origin = cam_pos + offset
                    dof_dir = normalize(focal_point - dof_origin)

                    color = color + trace_scene(
                        dof_origin, dof_dir,
                        s1_center, s1_radius, s1_reflectivity, s1_color,
                        s2_center, s2_radius, s2_color,
                        plane_point, plane_normal, checker_scale, checker_c1, checker_c2,
                        light_pos, light_intensity,
                        ambient, background, max_depth,
                        fog_density, fog_color
                    )

                frame[y, x] = color / dof_samples
            else:
                frame[y, x] = trace_scene(
                    cam_pos, direction,
                    s1_center, s1_radius, s1_reflectivity, s1_color,
                    s2_center, s2_radius, s2_color,
                    plane_point, plane_normal, checker_scale, checker_c1, checker_c2,
                    light_pos, light_intensity,
                    ambient, background, max_depth,
                    fog_density, fog_color
                )

    return frame


class FastRaytracer:
    """High-performance raytracer using Numba JIT compilation."""

    def __init__(
        self,
        max_depth: int = 2,
        fog_density: float = 0.0,
        fog_color: Tuple[float, float, float] = (0.5, 0.6, 0.8),
        dof_aperture: float = 0.0,
        dof_focal_dist: float = 4.0,
        dof_samples: int = 4,
        soft_shadows: bool = False,
        shadow_samples: int = 4,
        light_radius: float = 0.3
    ):
        self.max_depth = max_depth
        self.fog_density = fog_density
        self.fog_color = np.array(fog_color)
        self.dof_aperture = dof_aperture
        self.dof_focal_dist = dof_focal_dist
        self.dof_samples = dof_samples if dof_aperture > 0 else 1
        self.soft_shadows = soft_shadows
        self.shadow_samples = shadow_samples if soft_shadows else 1
        self.light_radius = light_radius

        # Scene objects (set by setup_scene)
        self.s1_center = np.array([0.0, 0.8, 0.0])
        self.s1_radius = 0.8
        self.s1_reflectivity = 0.85
        self.s1_color = np.array([0.95, 0.95, 1.0])

        self.s2_center = np.array([-1.5, 0.5, 1.2])
        self.s2_radius = 0.5
        self.s2_color = np.array([0.8, 0.2, 0.2])

        self.plane_point = np.array([0.0, 0.0, 0.0])
        self.plane_normal = np.array([0.0, 1.0, 0.0])
        self.checker_scale = 1.0
        self.checker_c1 = np.array([0.9, 0.9, 0.9])
        self.checker_c2 = np.array([0.1, 0.1, 0.1])

        self.light_pos = np.array([2.0, 3.0, 2.0])
        self.light_intensity = 1.3

        self.ambient = 0.1
        self.background = np.array([0.05, 0.05, 0.15])

    def update_light(self, position: np.ndarray):
        """Update light position."""
        self.light_pos = position

    def render_frame(self, camera, width: int, height: int) -> np.ndarray:
        """Render a frame using the fast JIT-compiled renderer."""
        return render_frame_fast(
            width, height,
            camera.position,
            camera.forward,
            camera.right,
            camera.up,
            camera.half_width,
            camera.half_height,
            self.s1_center, self.s1_radius, self.s1_reflectivity, self.s1_color,
            self.s2_center, self.s2_radius, self.s2_color,
            self.plane_point, self.plane_normal, self.checker_scale, self.checker_c1, self.checker_c2,
            self.light_pos, self.light_intensity,
            self.ambient, self.background, self.max_depth,
            self.fog_density, self.fog_color,
            self.dof_aperture, self.dof_focal_dist, self.dof_samples,
            self.shadow_samples, self.light_radius
        )


def warmup():
    """Warm up JIT compilation by running a small render."""
    from terminalray.raytracer import Camera
    from terminalray.math3d import vec3

    cam = Camera(
        position=vec3(0, 2, 4),
        look_at=vec3(0, 0.5, 0),
        fov=60.0
    )

    rt = FastRaytracer()
    _ = rt.render_frame(cam, 4, 4)
