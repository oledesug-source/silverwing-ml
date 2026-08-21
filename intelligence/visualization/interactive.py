"""Interactive terminal visualizations for Silverwing-ML."""

from dataclasses import dataclass, field


class ProgressBar:
    def __init__(self, total: int, description: str = "", width: int = 40):
        self.total = max(1, total)
        self.description = description
        self.width = width
        self.current = 0

    def update(self, n: int):
        self.current = min(self.total, self.current + n)

    def finish(self):
        self.current = self.total

    def render(self) -> str:
        pct = self.current / self.total
        filled = int(pct * self.width)
        bar = chr(9608) * filled + "░" * (self.width - filled)
        label = f"{self.description} " if self.description else ""
        return f"{label}{bar} {pct*100:.0f}% [{self.current}/{self.total}]"


class LiveChart:
    def __init__(self, max_points: int = 50, width: int = 60, height: int = 15):
        self.max_points = max_points
        self.width = width
        self.height = height
        self._data = []

    def add_point(self, value: float):
        self._data.append(value)
        if len(self._data) > self.max_points:
            self._data = self._data[-self.max_points:]

    def render(self) -> str:
        if not self._data:
            return " " * self.width

        data = self._data
        min_val = min(data)
        max_val = max(data)
        if min_val == max_val:
            max_val = min_val + 1

        blocks = " ▁▂▃▄▅▆▇█"
        n = len(data)
        step = max(1, n // self.width)
        sampled = [data[i] for i in range(0, n, step)][:self.width]

        result = ""
        for v in sampled:
            t = (v - min_val) / (max_val - min_val)
            idx = int(t * (len(blocks) - 1))
            idx = max(0, min(len(blocks) - 1, idx))
            result += blocks[idx]

        min_label = f"{min_val:.1f}"
        max_label = f"{max_val:.1f}"
        legend = f" {max_label}\n{result}\n{min_label}{' ' * (len(result) - len(min_label))}"

        return legend


class Spinner:
    DEFAULT_FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, frames: list = None):
        self.frames = frames or Spinner.DEFAULT_FRAMES
        self._index = 0

    def next_frame(self) -> str:
        frame = self.frames[self._index % len(self.frames)]
        self._index += 1
        return frame


@dataclass
class TablePrinter:
    headers: list = field(default_factory=list)
    widths: list = field(default_factory=list)
    _rows: list = field(default_factory=list, repr=False)
    _sort_col: int = field(default=-1, repr=False)
    _sort_rev: bool = field(default=False, repr=False)

    def add_row(self, values: list):
        self._rows.append(list(values))

    def render(self) -> str:
        rows = list(self._rows)
        if self._sort_col >= 0 and self._sort_col < (len(self.headers) if self.headers else 0):
            rows.sort(key=lambda r: r[self._sort_col] if self._sort_col < len(r) else "",
                       reverse=self._sort_rev)

        if not self.widths and self.headers:
            self.widths = [max(len(str(h)), 10) for h in self.headers]
        if not self.widths:
            return ""

        def fmt_row(values):
            parts = []
            for i, w in enumerate(self.widths):
                val = str(values[i]) if i < len(values) else ""
                parts.append(val.ljust(w))
            return "│ " + " │ ".join(parts) + " │"

        sep_inner = "┼".join("─" * (w + 2) for w in self.widths)
        sep_top = "┬".join("─" * (w + 2) for w in self.widths)
        sep_bottom = "┴".join("─" * (w + 2) for w in self.widths)

        lines = []
        lines.append("┌" + sep_top + "┐")
        if self.headers:
            lines.append(fmt_row(self.headers))
            lines.append("├" + sep_inner + "┤")
        for row in rows:
            lines.append(fmt_row(row))
        lines.append("└" + sep_bottom + "┘")

        return "\n".join(lines)

    def sort_by(self, column_index: int, reverse: bool = False):
        self._sort_col = column_index
        self._sort_rev = reverse

    def filter_rows(self, predicate):
        self._rows = [r for r in self._rows if predicate(r)]
