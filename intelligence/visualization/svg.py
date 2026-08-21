"""SVG generation utilities for Silverwing-ML visualization."""

import html as _html
import math


class SVGCanvas:
    def __init__(self, width: int, height: int, viewBox: str = None):
        self.width = width
        self.height = height
        self.viewBox = viewBox or f"0 0 {width} {height}"
        self._elements = []

    def _esc(self, text: str) -> str:
        return _html.escape(str(text))

    def rect(self, x, y, w, h, fill="black", stroke="none", rx=0):
        attrs = f'x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}"'
        if rx:
            attrs += f' rx="{rx}"'
        self._elements.append(f"  <rect {attrs}/>")

    def circle(self, cx, cy, r, fill="black", stroke="none"):
        self._elements.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}"/>')

    def ellipse(self, cx, cy, rx, ry, fill="black", stroke="none"):
        self._elements.append(f'  <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}"/>')

    def line(self, x1, y1, x2, y2, stroke="black", stroke_width=1):
        self._elements.append(
            f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def polyline(self, points, stroke="black", fill="none"):
        pts = " ".join(f"{p[0]},{p[1]}" for p in points)
        self._elements.append(f'  <polyline points="{pts}" stroke="{stroke}" fill="{fill}"/>')

    def polygon(self, points, fill="black", stroke="none"):
        pts = " ".join(f"{p[0]},{p[1]}" for p in points)
        self._elements.append(f'  <polygon points="{pts}" fill="{fill}" stroke="{stroke}"/>')

    def text(self, x, y, content, font_size=14, fill="black", anchor="start"):
        self._elements.append(
            f'  <text x="{x}" y="{y}" font-size="{font_size}" fill="{fill}" '
            f'text-anchor="{anchor}">{self._esc(content)}</text>'
        )

    def path(self, d, fill="none", stroke="black"):
        self._elements.append(f'  <path d="{d}" fill="{fill}" stroke="{stroke}"/>')

    def group(self, transform=None):
        if transform:
            self._elements.append(f'  <g transform="{transform}">')
        else:
            self._elements.append("  <g>")
        return self

    def end_group(self):
        self._elements.append("  </g>")

    def render(self) -> str:
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="{self.viewBox}">',
        ]
        parts.extend(self._elements)
        parts.append("</svg>")
        return "\n".join(parts)

    def save(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.render())


class SVGChart:
    def to_svg(self) -> SVGCanvas:
        raise NotImplementedError


class BarChartSVG(SVGChart):
    def __init__(self, data: dict, title: str = "", colors: list = None):
        self.data = dict(data)
        self.title = title
        self.colors = colors or ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    def to_svg(self) -> SVGCanvas:
        canvas = SVGCanvas(600, 400)
        canvas.rect(0, 0, 600, 400, fill="white")

        if self.title:
            canvas.text(300, 25, self.title, font_size=18, fill="#333", anchor="middle")

        items = list(self.data.items())
        if not items:
            return canvas

        max_val = max(abs(v) for _, v in items)
        if max_val == 0:
            max_val = 1

        top = 50
        bottom = 350
        left = 80
        right = 580
        bar_area_height = bottom - top
        bar_area_width = right - left
        bar_count = len(items)
        gap = bar_area_width / bar_count * 0.2
        bar_width = (bar_area_width - gap * bar_count) / bar_count

        for i, (label, value) in enumerate(items):
            x = left + i * (bar_width + gap) + gap / 2
            h = abs(value) / max_val * bar_area_height
            y = bottom - h
            color = self.colors[i % len(self.colors)]
            canvas.rect(x, y, bar_width, h, fill=color, stroke="#333")
            canvas.text(x + bar_width / 2, bottom + 15, label, font_size=10, fill="#333", anchor="middle")
            canvas.text(x + bar_width / 2, y - 5, f"{value:.1f}", font_size=9, fill="#333", anchor="middle")

        canvas.line(left, bottom, right, bottom, stroke="#333")
        canvas.line(left, top, left, bottom, stroke="#333")

        return canvas


class LineChartSVG(SVGChart):
    def __init__(self, data: list = None, title: str = "", colors: list = None):
        self._series = []
        if data:
            self._series.append({"data": list(data), "color": "#4e79a7"})
        self.title = title
        self.colors = colors or ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    def add_series(self, data: list, color: str = None):
        idx = len(self._series)
        self._series.append({"data": list(data), "color": color or self.colors[idx % len(self.colors)]})

    def to_svg(self) -> SVGCanvas:
        canvas = SVGCanvas(600, 400)
        canvas.rect(0, 0, 600, 400, fill="white")

        if self.title:
            canvas.text(300, 25, self.title, font_size=18, fill="#333", anchor="middle")

        if not self._series:
            return canvas

        all_vals = []
        for s in self._series:
            all_vals.extend(s["data"])
        if not all_vals:
            return canvas

        min_v, max_v = min(all_vals), max(all_vals)
        if min_v == max_v:
            max_v = min_v + 1

        left, right, top, bottom = 60, 580, 50, 350

        canvas.line(left, bottom, right, bottom, stroke="#999")
        canvas.line(left, top, left, bottom, stroke="#999")

        for i in range(5):
            y = top + (bottom - top) * i / 4
            val = max_v - (max_v - min_v) * i / 4
            canvas.line(left, y, left - 5, y, stroke="#999")
            canvas.text(left - 10, y + 4, f"{val:.1f}", font_size=9, fill="#666", anchor="end")

        for _si, s in enumerate(self._series):
            data = s["data"]
            color = s["color"]
            n = len(data)
            if n == 0:
                continue
            points = []
            for i, v in enumerate(data):
                x = left + (i / max(1, n - 1)) * (right - left)
                y = top + (1 - (v - min_v) / (max_v - min_v)) * (bottom - top)
                points.append((x, y))
            if len(points) > 1:
                canvas.polyline(points, stroke=color)
            for x, y in points:
                canvas.circle(x, y, 3, fill=color, stroke="white")

        return canvas


class PieChartSVG(SVGChart):
    def __init__(self, data: dict, title: str = ""):
        self.data = dict(data)
        self.title = title

    def to_svg(self) -> SVGCanvas:
        canvas = SVGCanvas(400, 400)
        canvas.rect(0, 0, 400, 400, fill="white")

        if self.title:
            canvas.text(200, 25, self.title, font_size=18, fill="#333", anchor="middle")

        total = sum(self.data.values())
        if total == 0:
            return canvas

        cx, cy, r = 200, 210, 140
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                   "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

        start_angle = -90
        for i, (_label, value) in enumerate(self.data.items()):
            sweep = value / total * 360
            end_angle = start_angle + sweep

            start_rad = math.radians(start_angle)
            end_rad = math.radians(end_angle)

            x1 = cx + r * math.cos(start_rad)
            y1 = cy + r * math.sin(start_rad)
            x2 = cx + r * math.cos(end_rad)
            y2 = cy + r * math.sin(end_rad)

            large_arc = 1 if sweep > 180 else 0
            color = colors[i % len(colors)]

            d = f"M {cx} {cy} L {x1:.1f} {y1:.1f} A {r} {r} 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z"
            canvas.path(d, fill=color, stroke="white")

            mid_angle = math.radians((start_angle + end_angle) / 2)
            label_r = r * 0.65
            lx = cx + label_r * math.cos(mid_angle)
            ly = cy + label_r * math.sin(mid_angle)
            pct = value / total * 100
            canvas.text(lx, ly, f"{pct:.1f}%", font_size=11, fill="white", anchor="middle")

            start_angle = end_angle

        y_offset = 370
        for i, (label, _value) in enumerate(self.data.items()):
            color = colors[i % len(colors)]
            x = 20 + (i % 5) * 100
            y = y_offset + (i // 5) * 18
            canvas.rect(x, y - 10, 10, 10, fill=color)
            canvas.text(x + 14, y, label, font_size=9, fill="#333")

        return canvas


class ScatterPlotSVG(SVGChart):
    def __init__(self, x: list = None, y: list = None, title: str = ""):
        self.x = list(x) if x else []
        self.y = list(y) if y else []
        self.title = title

    def to_svg(self) -> SVGCanvas:
        canvas = SVGCanvas(600, 400)
        canvas.rect(0, 0, 600, 400, fill="white")

        if self.title:
            canvas.text(300, 25, self.title, font_size=18, fill="#333", anchor="middle")

        if not self.x or not self.y:
            return canvas

        left, right, top, bottom = 60, 570, 40, 350
        min_x, max_x = min(self.x), max(self.x)
        min_y, max_y = min(self.y), max(self.y)
        if min_x == max_x:
            max_x = min_x + 1
        if min_y == max_y:
            max_y = min_y + 1

        canvas.line(left, bottom, right, bottom, stroke="#999")
        canvas.line(left, top, left, bottom, stroke="#999")

        for xi, yi in zip(self.x, self.y):
            px = left + (xi - min_x) / (max_x - min_x) * (right - left)
            py = bottom - (yi - min_y) / (max_y - min_y) * (bottom - top)
            canvas.circle(px, py, 4, fill="#4e79a7", stroke="#333")

        return canvas


class HistogramSVG(SVGChart):
    def __init__(self, data: list = None, bins: int = 10, title: str = ""):
        self.data = list(data) if data else []
        self.bins = bins
        self.title = title

    def to_svg(self) -> SVGCanvas:
        canvas = SVGCanvas(600, 400)
        canvas.rect(0, 0, 600, 400, fill="white")

        if self.title:
            canvas.text(300, 25, self.title, font_size=18, fill="#333", anchor="middle")

        if not self.data:
            return canvas

        min_val = min(self.data)
        max_val = max(self.data)
        if min_val == max_val:
            max_val = min_val + 1

        bin_size = (max_val - min_val) / self.bins
        counts = [0] * self.bins
        for v in self.data:
            idx = min(int((v - min_val) / bin_size), self.bins - 1)
            counts[idx] += 1

        max_count = max(counts) if counts else 1
        if max_count == 0:
            max_count = 1

        left, right, top, bottom = 60, 570, 50, 350
        bar_width = (right - left) / self.bins

        canvas.line(left, bottom, right, bottom, stroke="#999")
        canvas.line(left, top, left, bottom, stroke="#999")

        for i in range(self.bins):
            x = left + i * bar_width
            h = counts[i] / max_count * (bottom - top)
            canvas.rect(x + 1, bottom - h, bar_width - 2, h, fill="#4e79a7", stroke="#333")
            lo = min_val + i * bin_size
            canvas.text(x + bar_width / 2, bottom + 15, f"{lo:.0f}", font_size=8, fill="#666", anchor="middle")

        return canvas


class RadarChartSVG(SVGChart):
    def __init__(self, categories: dict, title: str = ""):
        self.categories = dict(categories)
        self.title = title

    def to_svg(self) -> SVGCanvas:
        canvas = SVGCanvas(400, 400)
        canvas.rect(0, 0, 400, 400, fill="white")

        if self.title:
            canvas.text(200, 20, self.title, font_size=18, fill="#333", anchor="middle")

        cx, cy, r = 200, 210, 140
        n = len(self.categories)
        if n < 3:
            return canvas

        items = list(self.categories.items())
        angles = [i * 2 * math.pi / n - math.pi / 2 for i in range(n)]

        for frac in [0.25, 0.5, 0.75, 1.0]:
            pts = []
            for angle in angles:
                px = cx + int(frac * r * math.cos(angle))
                py = cy + int(frac * r * math.sin(angle))
                pts.append((px, py))
            pts.append(pts[0])
            canvas.polyline(pts, stroke="#ccc")

        for angle in angles:
            canvas.line(cx, cy, cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle)), stroke="#ddd")

        data_pts = []
        for i, (name, val) in enumerate(items):
            val = max(0.0, min(1.0, val))
            px = cx + int(val * r * math.cos(angles[i]))
            py = cy + int(val * r * math.sin(angles[i]))
            data_pts.append((px, py))

            canvas.line(cx, cy, px, py, stroke="#4e79a7")
            canvas.circle(px, py, 4, fill="#4e79a7", stroke="white")

            lx = cx + int((r + 20) * math.cos(angles[i]))
            ly = cy + int((r + 20) * math.sin(angles[i]))
            canvas.text(lx, ly, name, font_size=11, fill="#333", anchor="middle")

        data_pts.append(data_pts[0])
        canvas.polygon(data_pts, fill="rgba(78,121,167,0.2)", stroke="#4e79a7")

        return canvas


class DashboardSVG:
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self._charts = []
        self._layout = []

    def add_chart(self, chart: SVGChart, row: int, col: int, row_span: int = 1, col_span: int = 1):
        self._charts.append(chart)
        self._layout.append({
            "chart": chart,
            "row": row,
            "col": col,
            "row_span": row_span,
            "col_span": col_span,
        })

    def render(self) -> str:
        canvas = SVGCanvas(self.width, self.height)
        canvas.rect(0, 0, self.width, self.height, fill="#f5f5f5")

        max_row = max((l["row"] + l["row_span"] for l in self._layout), default=1)
        max_col = max((l["col"] + l["col_span"] for l in self._layout), default=1)
        cell_w = self.width / max_col
        cell_h = self.height / max_row

        for entry in self._layout:
            chart = entry["chart"]
            x = int(entry["col"] * cell_w)
            y = int(entry["row"] * cell_h)
            w = int(entry["col_span"] * cell_w)
            h = int(entry["row_span"] * cell_h)

            svg_str = chart.to_svg().render()
            canvas._elements.append(f'  <g transform="translate({x},{y})" clip-path="url(#clip)">')
            canvas._elements.append(f'    <svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
            inner = svg_str.split("\n")
            for line in inner:
                if line.strip():
                    canvas._elements.append(f"      {line}")
            canvas._elements.append("    </svg>")
            canvas._elements.append("  </g>")

        return canvas.render()

    def save(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.render())
