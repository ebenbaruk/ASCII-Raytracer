"""Tests for object intersections."""

import numpy as np
import pytest

from terminalray.math3d import vec3, Ray, normalize
from terminalray.objects import Sphere, Plane
from terminalray.materials import DiffuseMaterial


@pytest.fixture
def simple_material():
    return DiffuseMaterial(color=vec3(1, 0, 0))


class TestSphereIntersection:
    def test_ray_hits_sphere_front(self, simple_material):
        sphere = Sphere(center=vec3(0, 0, 5), radius=1.0, material=simple_material)
        ray = Ray(origin=vec3(0, 0, 0), direction=vec3(0, 0, 1))

        hit = sphere.intersect(ray)

        assert hit is not None
        assert hit.t == pytest.approx(4.0)  # Hits at z=4 (front of sphere)
        assert hit.point[2] == pytest.approx(4.0)
        assert hit.normal[2] == pytest.approx(-1.0)  # Normal points toward ray

    def test_ray_misses_sphere(self, simple_material):
        sphere = Sphere(center=vec3(0, 0, 5), radius=1.0, material=simple_material)
        ray = Ray(origin=vec3(5, 0, 0), direction=vec3(0, 0, 1))

        hit = sphere.intersect(ray)

        assert hit is None

    def test_ray_inside_sphere(self, simple_material):
        sphere = Sphere(center=vec3(0, 0, 0), radius=5.0, material=simple_material)
        ray = Ray(origin=vec3(0, 0, 0), direction=vec3(0, 0, 1))

        hit = sphere.intersect(ray)

        assert hit is not None
        assert hit.t == pytest.approx(5.0)  # Exits at z=5

    def test_sphere_behind_ray(self, simple_material):
        sphere = Sphere(center=vec3(0, 0, -5), radius=1.0, material=simple_material)
        ray = Ray(origin=vec3(0, 0, 0), direction=vec3(0, 0, 1))

        hit = sphere.intersect(ray)

        assert hit is None


class TestPlaneIntersection:
    def test_ray_hits_plane(self, simple_material):
        plane = Plane(point=vec3(0, 0, 0), normal=vec3(0, 1, 0), material=simple_material)
        ray = Ray(origin=vec3(0, 5, 0), direction=normalize(vec3(0, -1, 0)))

        hit = plane.intersect(ray)

        assert hit is not None
        assert hit.t == pytest.approx(5.0)
        assert hit.point[1] == pytest.approx(0.0)

    def test_ray_parallel_to_plane(self, simple_material):
        plane = Plane(point=vec3(0, 0, 0), normal=vec3(0, 1, 0), material=simple_material)
        ray = Ray(origin=vec3(0, 5, 0), direction=vec3(1, 0, 0))

        hit = plane.intersect(ray)

        assert hit is None

    def test_plane_behind_ray(self, simple_material):
        plane = Plane(point=vec3(0, 0, 0), normal=vec3(0, 1, 0), material=simple_material)
        ray = Ray(origin=vec3(0, 5, 0), direction=vec3(0, 1, 0))

        hit = plane.intersect(ray)

        assert hit is None

    def test_plane_normal_faces_ray(self, simple_material):
        plane = Plane(point=vec3(0, 0, 0), normal=vec3(0, 1, 0), material=simple_material)
        ray = Ray(origin=vec3(0, 5, 0), direction=normalize(vec3(0, -1, 0)))

        hit = plane.intersect(ray)

        assert hit is not None
        # Normal should face toward the ray
        assert hit.normal[1] == pytest.approx(1.0)
