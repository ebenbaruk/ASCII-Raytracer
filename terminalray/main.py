"""Main entry point for TerminalRay ASCII raytracer."""

import argparse
import sys
import time
import numpy as np

from terminalray.math3d import vec3
from terminalray.objects import Sphere, Plane
from terminalray.materials import DiffuseMaterial, MirrorMaterial, CheckerMaterial
from terminalray.scene import Scene, Light
from terminalray.raytracer import Camera, Raytracer
from terminalray.ascii import (
    frame_to_ascii, frame_to_color, frame_to_halfblock,
    get_palette, PALETTES, RESET_COLOR
)

# ANSI escape codes
CLEAR_SCREEN = "\x1b[2J\x1b[H"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"


def create_demo_scene() -> Scene:
    """Create the demo scene with reflective sphere, diffuse sphere, and checkerboard floor."""
    scene = Scene(ambient=0.1)

    # Checkerboard floor
    floor_material = CheckerMaterial(
        color1=vec3(0.9, 0.9, 0.9),
        color2=vec3(0.1, 0.1, 0.1),
        scale=1.0,
        shininess=8.0,
        specular=0.2
    )
    floor = Plane(
        point=vec3(0, 0, 0),
        normal=vec3(0, 1, 0),
        material=floor_material
    )
    scene.add_object(floor)

    # Mirror sphere (center)
    mirror_material = MirrorMaterial(
        reflectivity=0.85,
        color=vec3(0.95, 0.95, 1.0),
        shininess=128.0,
        specular=1.0
    )
    mirror_sphere = Sphere(
        center=vec3(0, 0.8, 0),
        radius=0.8,
        material=mirror_material
    )
    scene.add_object(mirror_sphere)

    # Diffuse sphere (offset)
    diffuse_material = DiffuseMaterial(
        color=vec3(0.8, 0.2, 0.2),
        shininess=32.0,
        specular=0.4
    )
    diffuse_sphere = Sphere(
        center=vec3(-1.5, 0.5, 1.2),
        radius=0.5,
        material=diffuse_material
    )
    scene.add_object(diffuse_sphere)

    return scene


def create_camera(t: float, radius: float = 4.0, aspect_ratio: float = 2.0) -> Camera:
    """Create an orbiting camera."""
    angle = t * 0.3  # Slow rotation
    x = radius * np.cos(angle)
    z = radius * np.sin(angle)
    y = 2.0

    return Camera(
        position=vec3(x, y, z),
        look_at=vec3(0, 0.5, 0),
        fov=60.0,
        aspect_ratio=aspect_ratio
    )


def update_light(scene: Scene, t: float) -> None:
    """Update/create the moving light in the scene."""
    angle = t * 0.7
    x = 2.5 * np.cos(angle)
    z = 2.5 * np.sin(angle)
    y = 3.0

    light = Light(
        position=vec3(x, y, z),
        intensity=1.3,
        color=vec3(1.0, 0.98, 0.95)
    )

    scene.lights.clear()
    scene.add_light(light)


def get_light_pos(t: float) -> np.ndarray:
    """Get light position for fast renderer."""
    angle = t * 0.7
    return np.array([2.5 * np.cos(angle), 3.0, 2.5 * np.sin(angle)])


def run_animation_standard(
    width: int,
    height: int,
    fps: float,
    palette: str,
    reflections: int,
    gamma: float,
    color_mode: bool = False,
    halfblock: bool = False
) -> None:
    """Run animation with standard (non-JIT) raytracer."""
    scene = create_demo_scene()
    raytracer = Raytracer(scene, max_depth=reflections)
    frame_time = 1.0 / fps

    # Adjust aspect ratio for halfblock mode
    aspect = 1.0 if halfblock else 2.0
    render_height = height * 2 if halfblock else height

    print(HIDE_CURSOR, end='', flush=True)

    try:
        start_time = time.perf_counter()
        frame_count = 0

        while True:
            frame_start = time.perf_counter()
            t = frame_start - start_time

            camera = create_camera(t, aspect_ratio=aspect)
            update_light(scene, t)

            frame = raytracer.render_frame(camera, width, render_height)

            if halfblock:
                output = frame_to_halfblock(frame, gamma)
            elif color_mode:
                output = frame_to_color(frame, palette, gamma)
            else:
                output = frame_to_ascii(frame, palette, gamma)

            print(CLEAR_SCREEN + output, end='', flush=True)

            frame_count += 1
            elapsed = time.perf_counter() - frame_start
            actual_fps = 1.0 / elapsed if elapsed > 0 else 0
            mode_str = "HALFBLOCK" if halfblock else ("COLOR" if color_mode else "ASCII")
            info = f"\n{RESET_COLOR}[{width}x{height}] {mode_str} | Frame: {frame_count} | Render: {elapsed*1000:.0f}ms | FPS: {actual_fps:.1f}"
            print(info, end='', flush=True)

            remaining = frame_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        print(SHOW_CURSOR + RESET_COLOR)
        print("\nTerminalRay finished.")


def run_animation_fast(
    width: int,
    height: int,
    fps: float,
    palette: str,
    reflections: int,
    gamma: float,
    color_mode: bool = False,
    halfblock: bool = False,
    fog: float = 0.0,
    dof: float = 0.0,
    dof_samples: int = 4
) -> None:
    """Run animation with Numba JIT-compiled raytracer."""
    from terminalray.raytracer_fast import FastRaytracer, warmup

    print("Warming up JIT compiler...", end='', flush=True)
    warmup()
    print(" done!")

    raytracer = FastRaytracer(
        max_depth=reflections,
        fog_density=fog,
        fog_color=(0.4, 0.5, 0.7),
        dof_aperture=dof,
        dof_focal_dist=4.0,
        dof_samples=dof_samples
    )

    frame_time = 1.0 / fps
    aspect = 1.0 if halfblock else 2.0
    render_height = height * 2 if halfblock else height

    print(HIDE_CURSOR, end='', flush=True)

    try:
        start_time = time.perf_counter()
        frame_count = 0

        while True:
            frame_start = time.perf_counter()
            t = frame_start - start_time

            camera = create_camera(t, aspect_ratio=aspect)
            raytracer.update_light(get_light_pos(t))

            frame = raytracer.render_frame(camera, width, render_height)

            if halfblock:
                output = frame_to_halfblock(frame, gamma)
            elif color_mode:
                output = frame_to_color(frame, palette, gamma)
            else:
                output = frame_to_ascii(frame, palette, gamma)

            print(CLEAR_SCREEN + output, end='', flush=True)

            frame_count += 1
            elapsed = time.perf_counter() - frame_start
            actual_fps = 1.0 / elapsed if elapsed > 0 else 0

            effects = []
            if fog > 0:
                effects.append("FOG")
            if dof > 0:
                effects.append("DOF")
            effects_str = " + ".join(effects) if effects else ""

            mode_str = "HALFBLOCK" if halfblock else ("COLOR" if color_mode else "ASCII")
            info = f"\n{RESET_COLOR}[{width}x{height}] FAST {mode_str}"
            if effects_str:
                info += f" [{effects_str}]"
            info += f" | Frame: {frame_count} | Render: {elapsed*1000:.0f}ms | FPS: {actual_fps:.1f}"
            print(info, end='', flush=True)

            remaining = frame_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        print(SHOW_CURSOR + RESET_COLOR)
        print("\nTerminalRay finished.")


def render_still(
    output: str,
    width: int,
    height: int,
    palette: str,
    reflections: int,
    gamma: float,
    t: float = 0.0,
    color_mode: bool = False,
    halfblock: bool = False,
    fast: bool = False,
    fog: float = 0.0,
    dof: float = 0.0
) -> None:
    """Render a single still frame and save to file."""
    aspect = 1.0 if halfblock else 2.0
    render_height = height * 2 if halfblock else height

    if fast:
        from terminalray.raytracer_fast import FastRaytracer, warmup
        print("Warming up JIT compiler...", end='', flush=True)
        warmup()
        print(" done!")

        raytracer = FastRaytracer(
            max_depth=reflections,
            fog_density=fog,
            dof_aperture=dof,
            dof_samples=8 if dof > 0 else 1
        )
        camera = create_camera(t, aspect_ratio=aspect)
        raytracer.update_light(get_light_pos(t))

        print(f"Rendering {width}x{render_height} frame (FAST mode)...")
        start = time.perf_counter()
        frame = raytracer.render_frame(camera, width, render_height)
    else:
        scene = create_demo_scene()
        raytracer = Raytracer(scene, max_depth=reflections)
        camera = create_camera(t, aspect_ratio=aspect)
        update_light(scene, t)

        print(f"Rendering {width}x{render_height} frame...")
        start = time.perf_counter()
        frame = raytracer.render_frame(camera, width, render_height)

    elapsed = time.perf_counter() - start
    print(f"Render time: {elapsed:.2f}s")

    if halfblock:
        result = frame_to_halfblock(frame, gamma)
    elif color_mode:
        result = frame_to_color(frame, palette, gamma)
    else:
        result = frame_to_ascii(frame, palette, gamma)

    if output == '-':
        print(result + RESET_COLOR)
    else:
        with open(output, 'w') as f:
            f.write(result + RESET_COLOR + '\n')
        print(f"Saved to {output}")


def main():
    parser = argparse.ArgumentParser(
        description='TerminalRay - ASCII Raytracer with RGB Color Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m terminalray.main --demo                    # Basic ASCII demo
  python -m terminalray.main --demo --color            # Full RGB color!
  python -m terminalray.main --demo --halfblock        # High-res color mode
  python -m terminalray.main --demo --fast             # 10-50x faster with Numba
  python -m terminalray.main --demo --fast --fog 0.1   # Atmospheric fog
  python -m terminalray.main --demo --fast --dof 0.1   # Depth of field blur
  python -m terminalray.main --demo --fast --color --fog 0.08 --dof 0.05  # Full cinematic!

Palettes: ''' + ', '.join(PALETTES.keys())
    )

    parser.add_argument('--demo', action='store_true', help='Run animated demo')
    parser.add_argument('--still', action='store_true', help='Render still frame')
    parser.add_argument('--output', '-o', type=str, default='frame.txt', help='Output file (use - for stdout)')

    # Resolution
    parser.add_argument('--width', '-W', type=int, default=120, help='Width in chars (default: 120)')
    parser.add_argument('--height', '-H', type=int, default=50, help='Height in chars (default: 50)')
    parser.add_argument('--fps', type=float, default=30.0, help='Target FPS (default: 30)')

    # Rendering
    parser.add_argument('--palette', '-p', type=str, default='blocks', help='ASCII palette')
    parser.add_argument('--reflections', '-r', type=int, default=2, help='Reflection bounces (default: 2)')
    parser.add_argument('--gamma', '-g', type=float, default=2.2, help='Gamma correction')

    # Color modes
    parser.add_argument('--color', '-c', action='store_true', help='Enable full RGB color output')
    parser.add_argument('--halfblock', action='store_true', help='Use half-block chars for 2x vertical resolution')

    # Performance
    parser.add_argument('--fast', '-f', action='store_true', help='Use Numba JIT for 10-50x speedup')

    # Effects (fast mode only)
    parser.add_argument('--fog', type=float, default=0.0, help='Fog density (0.05-0.2 recommended)')
    parser.add_argument('--dof', type=float, default=0.0, help='Depth of field aperture (0.05-0.15 recommended)')
    parser.add_argument('--dof-samples', type=int, default=4, help='DOF sample count (default: 4)')

    parser.add_argument('--time', '-t', type=float, default=0.0, help='Time for still frame')

    args = parser.parse_args()
    palette = get_palette(args.palette)

    # Halfblock implies color
    color_mode = args.color or args.halfblock

    # Effects require fast mode
    if (args.fog > 0 or args.dof > 0) and not args.fast:
        print("Note: --fog and --dof require --fast mode. Enabling fast mode.")
        args.fast = True

    if args.demo or (not args.still):
        if args.fast:
            run_animation_fast(
                width=args.width,
                height=args.height,
                fps=args.fps,
                palette=palette,
                reflections=args.reflections,
                gamma=args.gamma,
                color_mode=color_mode,
                halfblock=args.halfblock,
                fog=args.fog,
                dof=args.dof,
                dof_samples=args.dof_samples
            )
        else:
            run_animation_standard(
                width=args.width,
                height=args.height,
                fps=args.fps,
                palette=palette,
                reflections=args.reflections,
                gamma=args.gamma,
                color_mode=color_mode,
                halfblock=args.halfblock
            )
    elif args.still:
        render_still(
            output=args.output,
            width=args.width,
            height=args.height,
            palette=palette,
            reflections=args.reflections,
            gamma=args.gamma,
            t=args.time,
            color_mode=color_mode,
            halfblock=args.halfblock,
            fast=args.fast,
            fog=args.fog,
            dof=args.dof
        )


if __name__ == '__main__':
    main()
