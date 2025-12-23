# TerminalRay

**A real-time ASCII raytracer that renders stunning 3D scenes directly in your terminal.**

```
                        ████████
                   ████████████████
                ██████████████████████
              ████████████  ████████████
            ████████████      ████████████
           ███████████   ▄▄▄▄   ███████████
          ██████████   ▄██████▄   ██████████
         ██████████   ████████▓▓   ██████████
        ██████████   ██████████▓▓   ██████████
        █████████   ████████████▓▓   █████████
        █████████   ████████████▓▓   █████████
        ██████████   ██████████▓▓   ██████████
         ██████████   ████████▓▓   ██████████
          ██████████   ▀██████▀   ██████████
     ░░░░░░███████████   ▀▀▀▀   ███████████░░░░░░
   ░░▒▒░░▒▒░░████████████████████████████░░▒▒░░▒▒░░
  ░░▒▒▓▓▒▒░░▒▒░░████████████████████████░░▒▒░░▒▒▓▓▒▒░░
 ░░▒▒▓▓██▓▓▒▒░░▒▒░░░░░░░░░░░░░░░░░░░░░░▒▒░░▒▒▓▓██▓▓▒▒░░
░░▒▒▓▓████▓▓▒▒░░▒▒▓▓▒▒░░▒▒▓▓▒▒░░▒▒▓▓▒▒░░▒▒▓▓████▓▓▒▒░░
```

## Features

- **Real-time 3D Rendering** — Watch a reflective chrome sphere orbit over a checkerboard floor
- **Full RGB Color** — 24-bit true color output using ANSI escape codes
- **Numba JIT Acceleration** — 10-50x performance boost with parallel rendering
- **Cinematic Effects** — Fog, depth of field, soft shadows, and specular highlights
- **Multiple Render Modes** — ASCII art, colored blocks, or high-res half-block characters
- **Configurable** — Resolution, FPS, palettes, reflection depth, and more

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/ASCII-Raytracer.git
cd ASCII-Raytracer
pip install -r requirements.txt

# Run the demo
python -m terminalray.main --demo
```

## Installation

### Requirements
- Python 3.11+
- NumPy
- Numba (for fast mode)

```bash
pip install numpy numba
```

## Usage

### Basic Demo
```bash
# Classic ASCII rendering
python -m terminalray.main --demo

# Full RGB color mode
python -m terminalray.main --demo --color

# High-resolution with half-block characters
python -m terminalray.main --demo --halfblock
```

### Fast Mode (Recommended)
```bash
# 10-50x faster with Numba JIT compilation
python -m terminalray.main --demo --fast

# Fast + color
python -m terminalray.main --demo --fast --color
```

### Cinematic Effects
```bash
# Atmospheric fog
python -m terminalray.main --demo --fast --fog 0.1

# Depth of field blur
python -m terminalray.main --demo --fast --dof 0.1

# Full cinematic experience
python -m terminalray.main --demo --fast --halfblock --fog 0.08 --dof 0.05
```

### Still Frame Export
```bash
# Render a single frame to file
python -m terminalray.main --still --output frame.txt

# High-quality colored export
python -m terminalray.main --still --fast --color --width 160 --height 80 --output -
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--demo` | Run animated demo scene | — |
| `--still` | Render single frame | — |
| `--fast`, `-f` | Enable Numba JIT (10-50x faster) | off |
| `--color`, `-c` | Full RGB color output | off |
| `--halfblock` | Half-block chars for 2x vertical resolution | off |
| `--width`, `-W` | Width in characters | 120 |
| `--height`, `-H` | Height in characters | 50 |
| `--fps` | Target frames per second | 30 |
| `--reflections`, `-r` | Reflection bounce depth (0-3) | 2 |
| `--fog` | Fog density (0.05-0.2 recommended) | 0 |
| `--dof` | Depth of field aperture (0.05-0.15) | 0 |
| `--palette`, `-p` | ASCII palette preset | blocks |
| `--gamma`, `-g` | Gamma correction | 2.2 |
| `--output`, `-o` | Output file for still frames | frame.txt |

### Palette Presets

| Name | Characters | Best For |
|------|------------|----------|
| `classic` | ` .:-=+*#%@` | Traditional ASCII art |
| `blocks` | ` ░▒▓█` | Block-based rendering |
| `dense` | 70 characters | Maximum detail |
| `simple` | ` .oO@` | Minimal, clean look |
| `dots` | ` .:oO@` | Dot-based shading |

## How It Works

TerminalRay implements a complete raytracing pipeline:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Camera    │────▶│  Ray Cast   │────▶│ Intersection│
│  Generation │     │  Per Pixel  │     │   Testing   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   ASCII     │◀────│   Shading   │◀────│  Material   │
│  Conversion │     │ (Blinn-Phong)│     │  Sampling   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Rendering Pipeline

1. **Ray Generation** — Camera rays through each pixel with configurable FOV
2. **Intersection Testing** — Ray-sphere and ray-plane intersection
3. **Shading** — Blinn-Phong lighting with ambient, diffuse, and specular terms
4. **Shadows** — Shadow rays cast toward light sources
5. **Reflections** — Recursive ray bounces for mirror materials
6. **Effects** — Fog attenuation, depth of field sampling
7. **Output** — Luminance-to-ASCII mapping or RGB color codes

### Scene Objects

- **Mirror Sphere** — Reflective chrome orb at scene center
- **Diffuse Sphere** — Matte red sphere offset from center
- **Checkerboard Plane** — Infinite floor with alternating tiles
- **Point Light** — Orbiting light source with shadows

## Project Structure

```
terminalray/
├── main.py              # CLI entry point and animation loop
├── raytracer.py         # Core raytracing engine
├── raytracer_fast.py    # Numba JIT-optimized renderer
├── scene.py             # Scene and light definitions
├── objects.py           # Sphere and plane primitives
├── materials.py         # Material definitions
├── math3d.py            # Vector math utilities
├── ascii.py             # ASCII and color conversion
└── tests/
    ├── test_math3d.py
    └── test_intersections.py
```

## Performance

| Mode | Resolution | Typical FPS | Notes |
|------|------------|-------------|-------|
| Standard | 120×50 | 2-5 | Pure Python/NumPy |
| Fast | 120×50 | 30-60 | Numba JIT, parallel |
| Fast + Effects | 120×50 | 15-30 | With fog/DOF |
| Fast + Halfblock | 120×100 | 20-40 | 2x vertical res |

*Benchmarked on Apple M1. First frame includes JIT warmup.*

## Examples

### Recommended Configurations

```bash
# Best visual quality (slower)
python -m terminalray.main --demo --fast --halfblock --fog 0.06 --dof 0.04 --reflections 3

# Balanced performance/quality
python -m terminalray.main --demo --fast --color --fog 0.08

# Maximum performance
python -m terminalray.main --demo --fast --width 80 --height 30

# Retro ASCII aesthetic
python -m terminalray.main --demo --palette classic --width 100 --height 40
```

## Running Tests

```bash
# Run all tests
python -m pytest terminalray/tests/ -v

# Run with coverage
python -m pytest terminalray/tests/ --cov=terminalray
```

## Technical Details

### Rendering Equation

The raytracer uses the Blinn-Phong reflection model:

```
L = ambient + (diffuse + specular) * shadow + reflection * reflectivity
```

Where:
- `diffuse = max(N·L, 0) * color * intensity`
- `specular = (N·H)^shininess * intensity`
- `shadow = 0 if occluded, 1 otherwise`



---

<p align="center">
  <i>Made with ◾ and pixels</i>
</p>
