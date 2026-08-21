"""Text-based chart rendering for Silverwing-ML visualization."""

import math


class BarChart:
    def __init__(self, data: dict, title: str = "", width: int = 60, orientation: str = "horizontal"):
        self.data = dict(data)
        self.title = title
        self.width = width
        self.orientation = orientation
        self._series = []

    def add_series(self, name: str, values: dict):
        self._series.append((name, dict(values)))

    def render(self) -> str:
        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("=" * len(self.title))

        all_data = [dict(self.data)]
        for _name, vals in self._series:
            all_data.append(vals)

        all_keys = list(self.data.keys())
        for _, vals in self._series:
            for k in vals:
                if k not in all_keys:
                    all_keys.append(k)

        max_val = 0.0
        for d in all_data:
            for v in d.values():
                if abs(v) > max_val:
                    max_val = abs(v)
        if max_val == 0:
            max_val = 1.0

        if self.orientation == "horizontal":
            max_label_len = max((len(k) for k in all_keys), default=0)
            for key in all_keys:
                row = f"{key:>{max_label_len}} | "
                if len(all_data) == 1:
                    val = all_data[0].get(key, 0)
                    bar_len = int((abs(val) / max_val) * self.width)
                    bar = chr(9608) * bar_len
                    row += bar + f" {val}"
                else:
                    for i, (_sname, sdata) in enumerate(
                        [(None, all_data[0])] + self._series
                    ):
                        val = sdata.get(key, 0)
                        bar_len = int((abs(val) / max_val) * self.width)
                        marker = ["#", "*", "+", "x", "o"][i % 5]
                        bar = marker * bar_len
                        row += bar + " "
                    row = row.rstrip()
                lines.append(row)
        else:
            bar_width = max(1, (self.width - len(all_keys)) // len(all_keys))
            for level in range(self.width, 0, -1):
                row = ""
                for key in all_keys:
                    val = all_data[0].get(key, 0)
                    bar_height = int((abs(val) / max_val) * self.width)
                    if level <= bar_height:
                        row += chr(9608) * bar_width
                    else:
                        row += " " * bar_width
                lines.append(row)
            labels_row = ""
            for key in all_keys:
                labels_row += key[:bar_width].center(bar_width)
            lines.append(labels_row)

        return "\n".join(lines)


class LineChart:
    def __init__(self, data: list = None, labels: list = None, title: str = "", width: int = 60, height: int = 20):
        self._series = []
        if data:
            self._series.append({"data": list(data), "label": ""})
        self.labels = labels
        self.title = title
        self.width = width
        self.height = height

    def add_series(self, data: list, label: str = ""):
        self._series.append({"data": list(data), "label": label})

    def render(self) -> str:
        if not self._series:
            return self.title if self.title else ""

        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("=" * len(self.title))

        all_values = []
        for s in self._series:
            all_values.extend(s["data"])

        if not all_values:
            return "\n".join(lines)

        min_v = min(all_values)
        max_v = max(all_values)
        if max_v == min_v:
            max_v = min_v + 1

        canvas = [[" " for _ in range(self.width)] for _ in range(self.height)]

        markers = ["*", "+", "x", "o", "#"]
        for si, s in enumerate(self._series):
            data = s["data"]
            marker = markers[si % len(markers)]
            n = len(data)
            for i, v in enumerate(data):
                col = int(i / max(1, n - 1) * (self.width - 1)) if n > 1 else self.width // 2
                row = int((v - min_v) / (max_v - min_v) * (self.height - 1))
                row = self.height - 1 - row
                if 0 <= row < self.height and 0 <= col < self.width:
                    canvas[row][col] = marker

        y_label_width = 10
        for r in range(self.height):
            val = max_v - (r / (self.height - 1)) * (max_v - min_v)
            label = f"{val:>9.1f}"
            row_str = label + "|"
            for c in range(self.width):
                row_str += canvas[r][c]
            lines.append(row_str)

        lines.append(" " * y_label_width + "+" + "-" * self.width)

        if self.labels:
            x_labels = ""
            for i in range(self.width):
                idx = int(i / max(1, self.width - 1) * (len(self.labels) - 1)) if self.width > 1 else 0
                if idx < len(self.labels) and i % max(1, self.width // len(self.labels)) == 0:
                    lbl = self.labels[idx]
                    x_labels += lbl[0] if lbl else " "
                else:
                    x_labels += " "
            lines.append(" " * y_label_width + " " + x_labels)

        return "\n".join(lines)


class PieChart:
    def __init__(self, data: dict = None, title: str = "", width: int = 40, height: int = 20):
        self.data = dict(data) if data else {}
        self.title = title
        self.width = width
        self.height = height
        self._symbols = ["#", "*", "+", "x", "o", "@", "%", "=", "~", "^"]

    def render(self) -> str:
        if not self.data:
            return self.title if self.title else ""

        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("=" * len(self.title))

        total = sum(self.data.values())
        if total == 0:
            total = 1.0

        cy = self.height // 2
        cx = self.width // 2
        ry = cy - 1
        rx = cx - 1

        canvas = [[" " for _ in range(self.width)] for _ in range(self.height)]

        angles = []
        cumulative = 0.0
        for v in self.data.values():
            angles.append(cumulative)
            cumulative += v / total
        angles.append(1.0)

        for row in range(self.height):
            for col in range(self.width):
                dy = (row - cy) / max(1, ry)
                dx = (col - cx) / max(1, rx)
                dist = dx * dx + dy * dy
                if dist <= 1.0:
                    angle = math.atan2(-(row - cy), col - cx) / (2 * math.pi)
                    if angle < 0:
                        angle += 1.0
                    for i in range(len(angles) - 1):
                        if angles[i] <= angle < angles[i + 1]:
                            canvas[row][col] = self._symbols[i % len(self._symbols)]
                            break

        for row in canvas:
            lines.append("".join(row))

        lines.append("")
        for i, (label, value) in enumerate(self.data.items()):
            pct = value / total * 100
            sym = self._symbols[i % len(self._symbols)]
            lines.append(f" {sym} {label}: {value} ({pct:.1f}%)")

        return "\n".join(lines)


class Histogram:
    def __init__(self, data: list = None, bins: int = 10, title: str = "", width: int = 60):
        self.data = list(data) if data else []
        self.bins = bins
        self.title = title
        self.width = width

    def render(self) -> str:
        if not self.data:
            return self.title if self.title else ""

        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("=" * len(self.title))

        min_val = min(self.data)
        max_val = max(self.data)
        if min_val == max_val:
            max_val = min_val + 1

        bin_size = (max_val - min_val) / self.bins
        counts = [0] * self.bins
        for v in self.data:
            idx = int((v - min_val) / bin_size)
            idx = min(idx, self.bins - 1)
            counts[idx] += 1

        max_count = max(counts) if counts else 1
        if max_count == 0:
            max_count = 1

        bar_max = self.width - 20
        for i in range(self.bins):
            lo = min_val + i * bin_size
            hi = lo + bin_size
            bar_len = int(counts[i] / max_count * bar_max)
            label = f"[{lo:7.2f},{hi:7.2f})"
            bar = chr(9608) * bar_len
            lines.append(f"{label} |{bar} {counts[i]}")

        return "\n".join(lines)


class ScatterPlot:
    def __init__(self, x: list = None, y: list = None, title: str = "", width: int = 60, height: int = 20):
        self.x = list(x) if x else []
        self.y = list(y) if y else []
        self.title = title
        self.width = width
        self.height = height

    def render(self) -> str:
        if not self.x or not self.y:
            return self.title if self.title else ""

        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("=" * len(self.title))

        min_x, max_x = min(self.x), max(self.x)
        min_y, max_y = min(self.y), max(self.y)
        if min_x == max_x:
            max_x = min_x + 1
        if min_y == max_y:
            max_y = min_y + 1

        canvas = [[" " for _ in range(self.width)] for _ in range(self.height)]

        for xi, yi in zip(self.x, self.y):
            col = int((xi - min_x) / (max_x - min_x) * (self.width - 1))
            row = self.height - 1 - int((yi - min_y) / (max_y - min_y) * (self.height - 1))
            row = max(0, min(self.height - 1, row))
            col = max(0, min(self.width - 1, col))
            canvas[row][col] = "*"

        y_label_w = 10
        for r in range(self.height):
            val = max_y - (r / max(1, self.height - 1)) * (max_y - min_y)
            lines.append(f"{val:>9.1f}|" + "".join(canvas[r]))

        lines.append(" " * y_label_w + "+" + "-" * self.width)
        x_label = f"{min_x:.1f}" + " " * (self.width - 10) + f"{max_x:.1f}"
        lines.append(" " * y_label_w + " " + x_label[:self.width])

        return "\n".join(lines)


class Sparkline:
    def __init__(self, data: list = None, width: int = 20):
        self.data = list(data) if data else []
        self.width = width

    def render(self) -> str:
        if not self.data:
            return ""

        blocks = " ▁▂▃▄▅▆▇█"
        min_val = min(self.data)
        max_val = max(self.data)
        if min_val == max_val:
            max_val = min_val + 1

        n = len(self.data)
        step = max(1, n // self.width)
        sampled = [self.data[i] for i in range(0, n, step)][:self.width]

        result = ""
        for v in sampled:
            t = (v - min_val) / (max_val - min_val)
            idx = int(t * (len(blocks) - 1))
            idx = max(0, min(len(blocks) - 1, idx))
            result += blocks[idx]

        return result


class Heatmap:
    def __init__(self, data: list = None, labels_x: list = None, labels_y: list = None, title: str = ""):
        self.data = [list(row) for row in data] if data else []
        self.labels_x = labels_x
        self.labels_y = labels_y
        self.title = title

    def render(self) -> str:
        if not self.data:
            return self.title if self.title else ""

        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("=" * len(self.title))

        all_vals = [v for row in self.data for v in row]
        min_val = min(all_vals)
        max_val = max(all_vals)
        if min_val == max_val:
            max_val = min_val + 1

        blocks = " ░▒▓█"

        if self.labels_x:
            header = "     "
            for lbl in self.labels_x:
                header += f"{lbl[:4]:>5}"
            lines.append(header)

        for r, row in enumerate(self.data):
            label = ""
            if self.labels_y and r < len(self.labels_y):
                label = f"{self.labels_y[r]:>4} "
            else:
                label = f"{r:>4} "
            row_str = label
            for v in row:
                t = (v - min_val) / (max_val - min_val)
                idx = int(t * (len(blocks) - 1))
                idx = max(0, min(len(blocks) - 1, idx))
                row_str += f"  {blocks[idx]}  "
            lines.append(row_str)

        return "\n".join(lines)


class BoxPlot:
    def __init__(self, data: dict = None, title: str = ""):
        self.data = dict(data) if data else {}
        self.title = title

    def _box_stats(self, values: list) -> dict:
        s = sorted(values)
        n = len(s)
        q1 = s[n // 4]
        q2 = s[n // 2]
        q3 = s[3 * n // 4]
        iqr = q3 - q1
        return {
            "min": s[0],
            "q1": q1,
            "median": q2,
            "q3": q3,
            "max": s[-1],
            "iqr": iqr,
            "lower_fence": max(s[0], q1 - 1.5 * iqr),
            "upper_fence": min(s[-1], q3 + 1.5 * iqr),
        }

    def render(self) -> str:
        if not self.data:
            return self.title if self.title else ""

        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("=" * len(self.title))

        all_vals = []
        for v in self.data.values():
            all_vals.extend(v)
        global_min = min(all_vals)
        global_max = max(all_vals)
        if global_min == global_max:
            global_max = global_min + 1

        plot_width = 50

        for label, values in self.data.items():
            stats = self._box_stats(values)
            scale = plot_width / (global_max - global_min)

            q1_pos = int((stats["q1"] - global_min) * scale)
            med_pos = int((stats["median"] - global_min) * scale)
            q3_pos = int((stats["q3"] - global_min) * scale)
            min_pos = int((stats["min"] - global_min) * scale)
            max_pos = int((stats["max"] - global_min) * scale)

            q1_pos = max(0, min(plot_width - 1, q1_pos))
            med_pos = max(0, min(plot_width - 1, med_pos))
            q3_pos = max(0, min(plot_width - 1, q3_pos))
            min_pos = max(0, min(plot_width - 1, min_pos))
            max_pos = max(0, min(plot_width - 1, max_pos))

            row = [" "] * plot_width
            for i in range(min_pos, q1_pos + 1):
                row[i] = "-"
            for i in range(q3_pos, max_pos + 1):
                row[i] = "-"
            for i in range(q1_pos, q3_pos + 1):
                row[i] = "█"
            row[med_pos] = "│"
            row[min_pos] = "└" if min_pos < q1_pos else "│"
            row[max_pos] = "┘" if max_pos > q3_pos else "│"

            lines.append(f"  {label:>12} " + "".join(row))

        return "\n".join(lines)


class WaterfallChart:
    def __init__(self, data: list = None, title: str = "", width: int = 60):
        self.data = list(data) if data else []
        self.title = title
        self.width = width

    def render(self) -> str:
        if not self.data:
            return self.title if self.title else ""

        lines = []
        if self.title:
            lines.append(self.title)
            lines.append("=" * len(self.title))

        cumulative = 0.0
        max_abs = 0.0
        entries = []
        for label, value in self.data:
            start = cumulative
            cumulative += value
            entries.append((label, value, start, cumulative))
            max_abs = max(max_abs, abs(cumulative), abs(start))

        if max_abs == 0:
            max_abs = 1.0

        bar_area = self.width - 20
        mid = bar_area // 2

        for label, value, start, end in entries:
            bar_start = int(start / max_abs * (bar_area // 2))
            bar_end = int(end / max_abs * (bar_area // 2))

            row = [" "] * bar_area
            lo = min(bar_start, bar_end)
            hi = max(bar_start, bar_end)
            lo = max(0, min(bar_area - 1, lo))
            hi = max(0, min(bar_area - 1, hi))

            if value >= 0:
                ch = "█"
            else:
                ch = "▒"
            for i in range(lo, hi + 1):
                row[i] = ch

            sign = "+" if value >= 0 else ""
            lines.append(f"{label:>15} |{''.join(row)}| {sign}{value:.1f}")

        lines.append(f"{'':>15} " + "+" + "-" * bar_area + "+")
        zero_pos = mid
        " " * (zero_pos + 1) + "0"
        lines.append(f"{'':>15} " + " " * bar_area)

        return "\n".join(lines)


class GanttChart:
    def __init__(self, tasks: list = None):
        self.tasks = list(tasks) if tasks else []

    def render(self) -> str:
        if not self.tasks:
            return ""

        max_end = 0
        for t in self.tasks:
            end = t.get("start", 0) + t.get("duration", 0)
            if end > max_end:
                max_end = end

        if max_end == 0:
            max_end = 1

        plot_width = 60
        name_width = 15

        lines = []
        header = " " * name_width + " "
        scale = plot_width / max_end
        for i in range(0, max_end + 1, max(1, max_end // 10)):
            pos = int(i * scale)
            header = header[:name_width + 1 + pos] + str(i) + header[name_width + 1 + pos + len(str(i)):]
        lines.append(header[:name_width + 1 + plot_width])
        lines.append("─" * (name_width + 1 + plot_width))

        symbols = "█▓▒░"
        for ti, t in enumerate(self.tasks):
            name = t.get("name", f"Task {ti + 1}")
            start = t.get("start", 0)
            duration = t.get("duration", 1)
            sym = symbols[ti % len(symbols)]

            row = [" "] * plot_width
            s_pos = int(start * scale)
            e_pos = int((start + duration) * scale)
            s_pos = max(0, min(plot_width - 1, s_pos))
            e_pos = max(0, min(plot_width, e_pos))

            for i in range(s_pos, e_pos):
                row[i] = sym

            lines.append(f"{name[:name_width]:>{name_width}} |{''.join(row)}|")

        return "\n".join(lines)


class RadarChart:
    def __init__(self, categories: dict = None):
        self.categories = dict(categories) if categories else {}

    def render(self) -> str:
        if not self.categories:
            return ""

        size = 21
        center = size // 2
        radius = center - 2

        canvas = [[" " for _ in range(size)] for _ in range(size)]

        n = len(self.categories)
        if n < 3:
            return str(dict(self.categories))

        categories = list(self.categories.items())
        angles = [i * 2 * math.pi / n - math.pi / 2 for i in range(n)]

        for r_frac in [0.25, 0.5, 0.75, 1.0]:
            for a in range(360):
                rad = math.radians(a)
                rr = int(r_frac * radius)
                px = int(center + rr * math.cos(rad))
                py = int(center - rr * math.sin(rad))
                if 0 <= px < size and 0 <= py < size:
                    if a % 45 == 0:
                        canvas[py][px] = "·"

        for i, angle in enumerate(angles):
            px = int(center + radius * math.cos(angle))
            py = int(center - radius * math.sin(angle))
            if 0 <= px < size and 0 <= py < size:
                canvas[py][px] = "●"

            label_x = int(center + (radius + 1) * math.cos(angle))
            label_y = int(center - (radius + 1) * math.sin(angle))
            name = categories[i][0][:5]
            if 0 <= label_x < size and 0 <= label_y < size:
                canvas[label_y][label_x] = name[0]

        prev_px, prev_py = None, None
        for i, (_name, val) in enumerate(categories):
            val = max(0.0, min(1.0, val))
            r = int(val * radius)
            px = int(center + r * math.cos(angles[i]))
            py = int(center - r * math.sin(angles[i]))
            px = max(0, min(size - 1, px))
            py = max(0, min(size - 1, py))

            if prev_px is not None:
                self._draw_line(canvas, prev_px, prev_py, px, py, "·")

            canvas[py][px] = "●"
            prev_px, prev_py = px, py

        if prev_px is not None:
            first_r = int(categories[0][1] * radius)
            first_px = int(center + first_r * math.cos(angles[0]))
            first_py = int(center - first_r * math.sin(angles[0]))
            first_px = max(0, min(size - 1, first_px))
            first_py = max(0, min(size - 1, first_py))
            self._draw_line(canvas, prev_px, prev_py, first_px, first_py, "·")

        lines = ["".join(row) for row in canvas]
        lines.append("")
        for name, val in categories:
            lines.append(f"  ● {name}: {val:.2f}")

        return "\n".join(lines)

    def _draw_line(self, canvas, x0, y0, x1, y1, ch):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steps = max(dx, dy)
        if steps == 0:
            return
        for s in range(1, steps):
            t = s / steps
            px = int(x0 + (x1 - x0) * t)
            py = int(y0 + (y1 - y0) * t)
            h = len(canvas)
            w = len(canvas[0]) if h > 0 else 0
            if 0 <= px < w and 0 <= py < h:
                if canvas[py][px] == " ":
                    canvas[py][px] = ch
