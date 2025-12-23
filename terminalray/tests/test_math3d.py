"""Tests for math3d module."""

import numpy as np
import pytest

from terminalray.math3d import (
    vec3, length, normalize, dot, cross, reflect, clamp, clamp_vec, Ray
)


class TestVec3:
    def test_vec3_creation(self):
        v = vec3(1, 2, 3)
        assert v[0] == 1
        assert v[1] == 2
        assert v[2] == 3

    def test_vec3_dtype(self):
        v = vec3(1, 2, 3)
        assert v.dtype == np.float64


class TestLength:
    def test_unit_vector(self):
        v = vec3(1, 0, 0)
        assert length(v) == pytest.approx(1.0)

    def test_zero_vector(self):
        v = vec3(0, 0, 0)
        assert length(v) == pytest.approx(0.0)

    def test_arbitrary_vector(self):
        v = vec3(3, 4, 0)
        assert length(v) == pytest.approx(5.0)


class TestNormalize:
    def test_unit_vector(self):
        v = vec3(5, 0, 0)
        n = normalize(v)
        assert n[0] == pytest.approx(1.0)
        assert n[1] == pytest.approx(0.0)
        assert n[2] == pytest.approx(0.0)

    def test_arbitrary_vector(self):
        v = vec3(3, 4, 0)
        n = normalize(v)
        assert length(n) == pytest.approx(1.0)
        assert n[0] == pytest.approx(0.6)
        assert n[1] == pytest.approx(0.8)

    def test_zero_vector(self):
        v = vec3(0, 0, 0)
        n = normalize(v)
        assert length(n) == pytest.approx(0.0)


class TestDot:
    def test_perpendicular_vectors(self):
        a = vec3(1, 0, 0)
        b = vec3(0, 1, 0)
        assert dot(a, b) == pytest.approx(0.0)

    def test_parallel_vectors(self):
        a = vec3(1, 0, 0)
        b = vec3(2, 0, 0)
        assert dot(a, b) == pytest.approx(2.0)

    def test_arbitrary_vectors(self):
        a = vec3(1, 2, 3)
        b = vec3(4, 5, 6)
        assert dot(a, b) == pytest.approx(32.0)  # 1*4 + 2*5 + 3*6


class TestCross:
    def test_x_cross_y(self):
        x = vec3(1, 0, 0)
        y = vec3(0, 1, 0)
        z = cross(x, y)
        assert z[0] == pytest.approx(0.0)
        assert z[1] == pytest.approx(0.0)
        assert z[2] == pytest.approx(1.0)


class TestReflect:
    def test_reflect_horizontal(self):
        # Ray coming down at 45 degrees
        v = vec3(1, -1, 0)
        n = vec3(0, 1, 0)
        r = reflect(normalize(v), n)
        # Should reflect up at 45 degrees
        expected = normalize(vec3(1, 1, 0))
        assert r[0] == pytest.approx(expected[0], abs=1e-6)
        assert r[1] == pytest.approx(expected[1], abs=1e-6)

    def test_reflect_straight_on(self):
        v = vec3(0, -1, 0)
        n = vec3(0, 1, 0)
        r = reflect(v, n)
        assert r[0] == pytest.approx(0.0)
        assert r[1] == pytest.approx(1.0)
        assert r[2] == pytest.approx(0.0)


class TestClamp:
    def test_clamp_below(self):
        assert clamp(-0.5) == 0.0

    def test_clamp_above(self):
        assert clamp(1.5) == 1.0

    def test_clamp_in_range(self):
        assert clamp(0.5) == 0.5


class TestClampVec:
    def test_clamp_vec(self):
        v = vec3(-0.5, 0.5, 1.5)
        c = clamp_vec(v)
        assert c[0] == pytest.approx(0.0)
        assert c[1] == pytest.approx(0.5)
        assert c[2] == pytest.approx(1.0)


class TestRay:
    def test_ray_at(self):
        ray = Ray(origin=vec3(0, 0, 0), direction=vec3(1, 0, 0))
        p = ray.at(5.0)
        assert p[0] == pytest.approx(5.0)
        assert p[1] == pytest.approx(0.0)
        assert p[2] == pytest.approx(0.0)
