"""ASCII conversion utilities for raytraced output."""

import numpy as np
from numpy.typing import NDArray

# Palette presets
PALETTES = {
    'classic': " .:-=+*#%@",
    'dense': " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    'blocks': " ░▒▓█",
    'simple': " .oO@",
    'dots': " .:oO@",
    'solid': "██████████",  # All solid blocks for color mode
}

DEFAULT_PALETTE = 'classic'

# ANSI escape codes for true color
RESET_COLOR = "\x1b[0m"


def rgb_to_ansi(r: int, g: int, b: int) -> str:
    """Convert RGB values (0-255) to ANSI 24-bit color escape code."""
    return f"\x1b[38;2;{r};{g};{b}m"


def rgb_to_ansi_bg(r: int, g: int, b: int) -> str:
    """Convert RGB values (0-255) to ANSI 24-bit background color."""
    return f"\x1b[48;2;{r};{g};{b}m"


def luminance(color: NDArray) -> float:
    """
    Convert RGB color to grayscale luminance.
    Uses standard luminance weights for human perception.
    """
    # Rec. 709 luminance coefficients
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def apply_gamma(value: float, gamma: float = 2.2) -> float:
    """Apply gamma correction to a luminance value."""
    return value ** (1.0 / gamma)


def to_ascii(lum: float, palette: str = None, gamma: float = 2.2) -> str:
    """
    Map a luminance value [0, 1] to an ASCII character.
    """
    if palette is None:
        palette = PALETTES[DEFAULT_PALETTE]

    # Apply gamma correction for better visual contrast
    lum = apply_gamma(max(0.0, min(1.0, lum)), gamma)

    # Map to palette index
    idx = int(lum * (len(palette) - 1))
    idx = max(0, min(len(palette) - 1, idx))

    return palette[idx]


def frame_to_ascii(
    frame: NDArray,
    palette: str = None,
    gamma: float = 2.2
) -> str:
    """
    Convert a rendered frame (H, W, 3) to an ASCII string.
    Returns a multiline string ready for terminal output.
    """
    if palette is None:
        palette = PALETTES[DEFAULT_PALETTE]

    height, width = frame.shape[:2]
    lines = []

    for y in range(height):
        line = []
        for x in range(width):
            lum = luminance(frame[y, x])
            char = to_ascii(lum, palette, gamma)
            line.append(char)
        lines.append(''.join(line))

    return '\n'.join(lines)


def get_palette(name: str) -> str:
    """Get a palette by name, or return the input if it's a custom palette."""
    if name in PALETTES:
        return PALETTES[name]
    return name


def frame_to_color(
    frame: NDArray,
    palette: str = None,
    gamma: float = 2.2,
    use_background: bool = False
) -> str:
    """
    Convert a rendered frame (H, W, 3) to a colored ASCII string.
    Uses ANSI 24-bit true color for full RGB rendering.

    Args:
        frame: RGB frame with values in [0, 1]
        palette: ASCII palette for character selection based on luminance
        gamma: Gamma correction value
        use_background: If True, use background color with spaces (denser look)
    """
    if palette is None:
        palette = PALETTES['blocks']

    height, width = frame.shape[:2]
    lines = []

    for y in range(height):
        line_parts = []
        prev_color = None

        for x in range(width):
            color = frame[y, x]

            # Apply gamma correction
            r = int(255 * min(1.0, max(0.0, color[0] ** (1.0 / gamma))))
            g = int(255 * min(1.0, max(0.0, color[1] ** (1.0 / gamma))))
            b = int(255 * min(1.0, max(0.0, color[2] ** (1.0 / gamma))))

            # Get character based on luminance
            lum = luminance(color)
            char = to_ascii(lum, palette, gamma)

            # Optimization: only emit color code if color changed
            current_color = (r, g, b)
            if current_color != prev_color:
                if use_background:
                    line_parts.append(rgb_to_ansi_bg(r, g, b) + " ")
                else:
                    line_parts.append(rgb_to_ansi(r, g, b) + char)
                prev_color = current_color
            else:
                if use_background:
                    line_parts.append(" ")
                else:
                    line_parts.append(char)

        lines.append(''.join(line_parts) + RESET_COLOR)

    return '\n'.join(lines)


def frame_to_halfblock(
    frame: NDArray,
    gamma: float = 2.2
) -> str:
    """
    Convert a rendered frame to colored output using half-block characters.
    This effectively doubles the vertical resolution by using ▀ (upper half block)
    with foreground color for top pixel and background color for bottom pixel.

    Args:
        frame: RGB frame with values in [0, 1], height should be even
    """
    height, width = frame.shape[:2]
    lines = []

    # Process two rows at a time
    for y in range(0, height - 1, 2):
        line_parts = []
        prev_fg = None
        prev_bg = None

        for x in range(width):
            # Top pixel (foreground)
            top = frame[y, x]
            r1 = int(255 * min(1.0, max(0.0, top[0] ** (1.0 / gamma))))
            g1 = int(255 * min(1.0, max(0.0, top[1] ** (1.0 / gamma))))
            b1 = int(255 * min(1.0, max(0.0, top[2] ** (1.0 / gamma))))

            # Bottom pixel (background)
            bot = frame[y + 1, x]
            r2 = int(255 * min(1.0, max(0.0, bot[0] ** (1.0 / gamma))))
            g2 = int(255 * min(1.0, max(0.0, bot[1] ** (1.0 / gamma))))
            b2 = int(255 * min(1.0, max(0.0, bot[2] ** (1.0 / gamma))))

            fg = (r1, g1, b1)
            bg = (r2, g2, b2)

            # Build escape sequence only when colors change
            codes = []
            if fg != prev_fg:
                codes.append(f"\x1b[38;2;{r1};{g1};{b1}m")
                prev_fg = fg
            if bg != prev_bg:
                codes.append(f"\x1b[48;2;{r2};{g2};{b2}m")
                prev_bg = bg

            line_parts.append(''.join(codes) + "▀")

        lines.append(''.join(line_parts) + RESET_COLOR)

    return '\n'.join(lines)
