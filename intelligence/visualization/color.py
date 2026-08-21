"""Color utilities for Silverwing-ML visualization."""

from dataclasses import dataclass


@dataclass
class Color:
    """RGB color representation."""

    r: int
    g: int
    b: int

    def __post_init__(self):
        self.r = max(0, min(255, int(self.r)))
        self.g = max(0, min(255, int(self.g)))
        self.b = max(0, min(255, int(self.b)))

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 3:
            hex_str = "".join(c * 2 for c in hex_str)
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return cls(r, g, b)

    @classmethod
    def from_name(cls, name: str) -> "Color":
        name_lower = name.lower().strip()
        if name_lower not in _CSS_COLORS:
            raise ValueError(f"Unknown color name: {name!r}")
        return cls(*_CSS_COLORS[name_lower])

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_rgb(self) -> tuple:
        return (self.r, self.g, self.b)

    def __add__(self, other: "Color") -> "Color":
        return Color(
            (self.r + other.r) // 2,
            (self.g + other.g) // 2,
            (self.b + other.b) // 2,
        )

    def __mul__(self, factor: float) -> "Color":
        return Color(
            int(self.r * factor),
            int(self.g * factor),
            int(self.b * factor),
        )

    def lerp(self, other: "Color", t: float) -> "Color":
        t = max(0.0, min(1.0, t))
        return Color(
            int(self.r + (other.r - self.r) * t),
            int(self.g + (other.g - self.g) * t),
            int(self.b + (other.b - self.b) * t),
        )

    def __repr__(self) -> str:
        return f"Color(r={self.r}, g={self.g}, b={self.b})"

    def __str__(self) -> str:
        return self.to_hex()


class ColorScale:
    """Gradient between multiple colors."""

    @staticmethod
    def gradient(colors: list, n_steps: int = 10) -> list:
        if len(colors) == 0:
            return []
        if len(colors) == 1:
            return list(colors) * n_steps
        if n_steps <= 0:
            return []
        result = []
        total_segments = len(colors) - 1
        for i in range(n_steps):
            t = i / max(1, n_steps - 1)
            segment = min(int(t * total_segments), total_segments - 1)
            local_t = (t * total_segments) - segment
            result.append(colors[segment].lerp(colors[segment + 1], local_t))
        return result

    @staticmethod
    def viridis(n: int = 10) -> list:
        control = [
            Color(68, 1, 84), Color(72, 35, 116), Color(64, 67, 135),
            Color(52, 94, 141), Color(41, 120, 142), Color(32, 144, 140),
            Color(34, 167, 132), Color(68, 190, 112), Color(122, 209, 81),
            Color(189, 222, 38), Color(253, 231, 37),
        ]
        return ColorScale.gradient(control, n)

    @staticmethod
    def plasma(n: int = 10) -> list:
        control = [
            Color(13, 8, 135), Color(75, 3, 161), Color(126, 3, 168),
            Color(168, 34, 150), Color(203, 70, 121), Color(229, 107, 93),
            Color(248, 148, 65), Color(253, 195, 40), Color(240, 249, 33),
        ]
        return ColorScale.gradient(control, n)

    @staticmethod
    def inferno(n: int = 10) -> list:
        control = [
            Color(0, 0, 4), Color(22, 11, 57), Color(66, 10, 104),
            Color(106, 23, 110), Color(147, 38, 100), Color(188, 55, 84),
            Color(221, 81, 58), Color(243, 118, 27), Color(250, 164, 19),
            Color(246, 215, 70), Color(252, 255, 164),
        ]
        return ColorScale.gradient(control, n)

    @staticmethod
    def magma(n: int = 10) -> list:
        control = [
            Color(0, 0, 4), Color(18, 13, 55), Color(51, 16, 104),
            Color(90, 18, 130), Color(132, 31, 143), Color(173, 52, 143),
            Color(211, 77, 135), Color(237, 113, 127), Color(251, 159, 146),
            Color(253, 208, 186), Color(252, 253, 191),
        ]
        return ColorScale.gradient(control, n)

    @staticmethod
    def coolwarm(n: int = 10) -> list:
        control = [
            Color(59, 76, 192), Color(98, 130, 234), Color(141, 176, 254),
            Color(184, 208, 249), Color(221, 221, 221),
            Color(245, 196, 173), Color(244, 154, 123), Color(222, 96, 77),
            Color(180, 4, 38),
        ]
        return ColorScale.gradient(control, n)

    @staticmethod
    def rainbow(n: int = 10) -> list:
        result = []
        for i in range(n):
            t = i / max(1, n - 1)
            h = t * 360
            result.append(ColorScale._hsv_to_rgb(h, 1.0, 1.0))
        return result

    @staticmethod
    def heat(n: int = 10) -> list:
        control = [
            Color(0, 0, 0), Color(128, 0, 0), Color(255, 0, 0),
            Color(255, 165, 0), Color(255, 255, 0), Color(255, 255, 255),
        ]
        return ColorScale.gradient(control, n)

    @staticmethod
    def grayscale(n: int = 10) -> list:
        control = [Color(0, 0, 0), Color(255, 255, 255)]
        return ColorScale.gradient(control, n)

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Color:
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        return Color(int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def colormap(value: float, min_val: float, max_val: float, scale: str = "viridis") -> Color:
    if max_val == min_val:
        t = 0.5
    else:
        t = (value - min_val) / (max_val - min_val)
    t = max(0.0, min(1.0, t))
    scales = {
        "viridis": ColorScale.viridis,
        "plasma": ColorScale.plasma,
        "inferno": ColorScale.inferno,
        "magma": ColorScale.magma,
        "coolwarm": ColorScale.coolwarm,
        "rainbow": ColorScale.rainbow,
        "heat": ColorScale.heat,
        "grayscale": ColorScale.grayscale,
    }
    if scale not in scales:
        scale = "viridis"
    colors = scales[scale](256)
    idx = int(t * (len(colors) - 1))
    return colors[idx]


_CSS_COLORS = {
    "aliceblue": (240, 248, 255), "antiquewhite": (250, 235, 215),
    "aqua": (0, 255, 255), "aquamarine": (127, 255, 212),
    "azure": (240, 255, 255), "beige": (245, 245, 220),
    "bisque": (255, 228, 196), "black": (0, 0, 0),
    "blanchedalmond": (255, 235, 205), "blue": (0, 0, 255),
    "blueviolet": (138, 43, 226), "brown": (165, 42, 42),
    "burlywood": (222, 184, 135), "cadetblue": (95, 158, 160),
    "chartreuse": (127, 255, 0), "chocolate": (210, 105, 30),
    "coral": (255, 127, 80), "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220), "crimson": (220, 20, 60),
    "cyan": (0, 255, 255), "darkblue": (0, 0, 139),
    "darkcyan": (0, 139, 139), "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169), "darkgreen": (0, 100, 0),
    "darkkhaki": (189, 183, 107), "darkmagenta": (139, 0, 139),
    "darkolivegreen": (85, 107, 47), "darkorange": (255, 140, 0),
    "darkorchid": (153, 50, 204), "darkred": (139, 0, 0),
    "darksalmon": (233, 150, 122), "darkseagreen": (143, 188, 143),
    "darkslateblue": (72, 61, 139), "darkslategray": (47, 79, 79),
    "darkturquoise": (0, 206, 209), "darkviolet": (148, 0, 211),
    "deeppink": (255, 20, 147), "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105), "dodgerblue": (30, 144, 255),
    "firebrick": (178, 34, 34), "floralwhite": (255, 250, 240),
    "forestgreen": (34, 139, 34), "fuchsia": (255, 0, 255),
    "gainsboro": (220, 220, 220), "ghostwhite": (248, 248, 255),
    "gold": (255, 215, 0), "goldenrod": (218, 165, 32),
    "gray": (128, 128, 128), "green": (0, 128, 0),
    "greenyellow": (173, 255, 47), "honeydew": (240, 255, 240),
    "hotpink": (255, 105, 180), "indianred": (205, 92, 92),
    "indigo": (75, 0, 130), "ivory": (255, 255, 240),
    "khaki": (240, 230, 140), "lavender": (230, 230, 250),
    "lavenderblush": (255, 240, 245), "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205), "lightblue": (173, 216, 230),
    "lightcoral": (240, 128, 128), "lightcyan": (224, 255, 255),
    "lightgoldenrodyellow": (250, 250, 210), "lightgray": (211, 211, 211),
    "lightgreen": (144, 238, 144), "lightpink": (255, 182, 193),
    "lightsalmon": (255, 160, 122), "lightseagreen": (32, 178, 170),
    "lightskyblue": (135, 206, 250), "lightslategray": (119, 136, 153),
    "lightsteelblue": (176, 196, 222), "lightyellow": (255, 255, 224),
    "lime": (0, 255, 0), "limegreen": (50, 205, 50),
    "linen": (250, 240, 230), "magenta": (255, 0, 255),
    "maroon": (128, 0, 0), "mediumaquamarine": (102, 205, 170),
    "mediumblue": (0, 0, 205), "mediumorchid": (186, 85, 211),
    "mediumpurple": (147, 112, 219), "mediumseagreen": (60, 179, 113),
    "mediumslateblue": (123, 104, 238), "mediumspringgreen": (0, 250, 154),
    "mediumturquoise": (72, 209, 204), "mediumvioletred": (199, 21, 133),
    "midnightblue": (25, 25, 112), "mintcream": (245, 255, 250),
    "mistyrose": (255, 228, 225), "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173), "navy": (0, 0, 128),
    "oldlace": (253, 245, 230), "olive": (128, 128, 0),
    "olivedrab": (107, 142, 35), "orange": (255, 165, 0),
    "orangered": (255, 69, 0), "orchid": (218, 112, 214),
    "palegoldenrod": (238, 232, 170), "palegreen": (152, 251, 152),
    "paleturquoise": (175, 238, 238), "palevioletred": (219, 112, 147),
    "papayawhip": (255, 239, 213), "peachpuff": (255, 218, 185),
    "peru": (205, 133, 63), "pink": (255, 192, 203),
    "plum": (221, 160, 221), "powderblue": (176, 224, 230),
    "purple": (128, 0, 128), "rebeccapurple": (102, 51, 153),
    "red": (255, 0, 0), "rosybrown": (188, 143, 143),
    "royalblue": (65, 105, 225), "saddlebrown": (139, 69, 19),
    "salmon": (250, 128, 114), "sandybrown": (244, 164, 96),
    "seagreen": (46, 139, 87), "seashell": (255, 245, 238),
    "sienna": (160, 82, 45), "silver": (192, 192, 192),
    "skyblue": (135, 206, 235), "slateblue": (106, 90, 205),
    "slategray": (112, 128, 144), "snow": (255, 250, 250),
    "springgreen": (0, 255, 127), "steelblue": (70, 130, 180),
    "tan": (210, 180, 140), "teal": (0, 128, 128),
    "thistle": (216, 191, 216), "tomato": (255, 99, 71),
    "turquoise": (64, 224, 208), "violet": (238, 130, 238),
    "wheat": (245, 222, 179), "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245), "yellow": (255, 255, 0),
    "yellowgreen": (154, 205, 50),
}
