"""Scene container with objects and lights."""

from dataclasses import dataclass, field
from typing import List, Optional

from terminalray.math3d import Vec3, vec3, Ray, Hit


@dataclass
class Light:
    """A point light source."""
    position: Vec3
    intensity: float = 1.0
    color: Vec3 = None

    def __post_init__(self):
        if self.color is None:
            self.color = vec3(1.0, 1.0, 1.0)


@dataclass
class Scene:
    """Container for all objects and lights in the scene."""
    objects: List = field(default_factory=list)
    lights: List[Light] = field(default_factory=list)
    background: Vec3 = None
    ambient: float = 0.1

    def __post_init__(self):
        if self.background is None:
            self.background = vec3(0.05, 0.05, 0.1)

    def add_object(self, obj) -> None:
        """Add an object to the scene."""
        self.objects.append(obj)

    def add_light(self, light: Light) -> None:
        """Add a light to the scene."""
        self.lights.append(light)

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """Find the closest intersection of a ray with scene objects."""
        closest_hit: Optional[Hit] = None
        closest_t = float('inf')

        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit is not None and hit.t < closest_t:
                closest_t = hit.t
                closest_hit = hit

        return closest_hit
