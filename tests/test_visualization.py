"""Comprehensive tests for intelligence.visualization module."""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.visualization.charts import (
    BarChart,
    BoxPlot,
    GanttChart,
    Heatmap,
    Histogram,
    LineChart,
    PieChart,
    RadarChart,
    ScatterPlot,
    Sparkline,
    WaterfallChart,
)
from intelligence.visualization.color import Color, ColorScale, colormap
from intelligence.visualization.html_charts import ChartJS, Dashboard, TableRenderer
from intelligence.visualization.interactive import LiveChart, ProgressBar, Spinner, TablePrinter
from intelligence.visualization.reports import Report, ReportTheme
from intelligence.visualization.svg import (
    BarChartSVG,
    DashboardSVG,
    HistogramSVG,
    LineChartSVG,
    PieChartSVG,
    RadarChartSVG,
    ScatterPlotSVG,
    SVGCanvas,
)


class TestBarChart:
    def test_render_basic(self):
        chart = BarChart({"A": 10, "B": 20, "C": 30}, title="Test Bar")
        result = chart.render()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test Bar" in result
        assert "A" in result

    def test_render_empty(self):
        chart = BarChart({}, title="Empty")
        result = chart.render()
        assert isinstance(result, str)

    def test_vertical_orientation(self):
        chart = BarChart({"X": 5, "Y": 15}, orientation="vertical")
        result = chart.render()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_add_series(self):
        chart = BarChart({"A": 10, "B": 20})
        chart.add_series("Series2", {"A": 5, "B": 25})
        result = chart.render()
        assert isinstance(result, str)
        assert len(result) > 0


class TestLineChart:
    def test_render_basic(self):
        chart = LineChart(data=[1, 3, 2, 5, 4], title="Test Line")
        result = chart.render()
        assert isinstance(result, str)
        assert "Test Line" in result

    def test_render_empty(self):
        chart = LineChart(title="Empty")
        result = chart.render()
        assert isinstance(result, str)

    def test_add_series(self):
        chart = LineChart(data=[1, 2, 3])
        chart.add_series([3, 2, 1], label="Inverse")
        result = chart.render()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_labels(self):
        chart = LineChart(data=[10, 20, 30], labels=["a", "b", "c"], title="Labeled")
        result = chart.render()
        assert isinstance(result, str)


class TestPieChart:
    def test_render_basic(self):
        chart = PieChart({"A": 30, "B": 70}, title="Test Pie")
        result = chart.render()
        assert isinstance(result, str)
        assert "A" in result
        assert "B" in result
        assert "%" in result

    def test_render_empty(self):
        chart = PieChart({})
        result = chart.render()
        assert isinstance(result, str)

    def test_legend_present(self):
        chart = PieChart({"X": 1, "Y": 2, "Z": 3})
        result = chart.render()
        assert "X" in result
        assert "Y" in result
        assert "Z" in result


class TestHistogram:
    def test_render_basic(self):
        chart = Histogram(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bins=5, title="Test Hist")
        result = chart.render()
        assert isinstance(result, str)
        assert "Test Hist" in result

    def test_render_empty(self):
        chart = Histogram(data=[])
        result = chart.render()
        assert isinstance(result, str)

    def test_single_value(self):
        chart = Histogram(data=[5, 5, 5, 5], bins=3)
        result = chart.render()
        assert isinstance(result, str)
        assert len(result) > 0


class TestScatterPlot:
    def test_render_basic(self):
        chart = ScatterPlot(x=[1, 2, 3], y=[4, 5, 6], title="Test Scatter")
        result = chart.render()
        assert isinstance(result, str)
        assert "Test Scatter" in result
        assert "*" in result

    def test_render_empty(self):
        chart = ScatterPlot()
        result = chart.render()
        assert isinstance(result, str)


class TestSparkline:
    def test_render_basic(self):
        chart = Sparkline(data=[1, 3, 2, 5, 4, 6], width=10)
        result = chart.render()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_empty(self):
        chart = Sparkline()
        result = chart.render()
        assert result == ""

    def test_single_value(self):
        chart = Sparkline(data=[42])
        result = chart.render()
        assert isinstance(result, str)
        assert len(result) > 0


class TestHeatmap:
    def test_render_basic(self):
        chart = Heatmap(data=[[1, 2], [3, 4]], title="Test Heatmap")
        result = chart.render()
        assert isinstance(result, str)
        assert "Test Heatmap" in result

    def test_with_labels(self):
        chart = Heatmap(
            data=[[10, 20], [30, 40]],
            labels_x=["Col1", "Col2"],
            labels_y=["Row1", "Row2"],
        )
        result = chart.render()
        assert "Col1" in result
        assert "Row1" in result


class TestBoxPlot:
    def test_render_basic(self):
        chart = BoxPlot({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]}, title="Test Box")
        result = chart.render()
        assert isinstance(result, str)
        assert "A" in result
        assert "B" in result

    def test_render_empty(self):
        chart = BoxPlot({})
        result = chart.render()
        assert isinstance(result, str)


class TestWaterfallChart:
    def test_render_basic(self):
        chart = WaterfallChart(
            [("Start", 100), ("Add", 50), ("Sub", -30), ("End", 20)],
            title="Test Waterfall",
        )
        result = chart.render()
        assert isinstance(result, str)
        assert "Test Waterfall" in result
        assert "Start" in result


class TestGanttChart:
    def test_render_basic(self):
        chart = GanttChart([
            {"name": "Task 1", "start": 0, "duration": 5},
            {"name": "Task 2", "start": 2, "duration": 4},
        ])
        result = chart.render()
        assert isinstance(result, str)
        assert "Task 1" in result
        assert "Task 2" in result


class TestRadarChart:
    def test_render_basic(self):
        chart = RadarChart({"Speed": 0.8, "Power": 0.6, "Defense": 0.4})
        result = chart.render()
        assert isinstance(result, str)
        assert "Speed" in result
        assert "Power" in result

    def test_render_empty(self):
        chart = RadarChart({})
        result = chart.render()
        assert result == ""


class TestColor:
    def test_from_hex(self):
        c = Color.from_hex("#ff0000")
        assert c.r == 255
        assert c.g == 0
        assert c.b == 0

    def test_from_hex_short(self):
        c = Color.from_hex("#f0f")
        assert c.r == 255
        assert c.g == 0
        assert c.b == 255

    def test_to_hex(self):
        c = Color(0, 128, 255)
        assert c.to_hex() == "#0080ff"

    def test_to_rgb(self):
        c = Color(10, 20, 30)
        assert c.to_rgb() == (10, 20, 30)

    def test_lerp(self):
        c1 = Color(0, 0, 0)
        c2 = Color(255, 255, 255)
        mid = c1.lerp(c2, 0.5)
        assert mid.r == 127
        assert mid.g == 127
        assert mid.b == 127

    def test_lerp_clamp(self):
        c1 = Color(0, 0, 0)
        c2 = Color(100, 100, 100)
        result = c1.lerp(c2, 2.0)
        assert result.r == 100

    def test_add_blend(self):
        c1 = Color(255, 0, 0)
        c2 = Color(0, 0, 255)
        blended = c1 + c2
        assert blended.r == 127
        assert blended.b == 127

    def test_mul_brighten(self):
        c = Color(100, 100, 100)
        bright = c * 2.0
        assert bright.r == 200

    def test_mul_darken(self):
        c = Color(200, 200, 200)
        dark = c * 0.5
        assert dark.r == 100

    def test_clamping(self):
        c = Color(300, -10, 128)
        assert c.r == 255
        assert c.g == 0
        assert c.b == 128

    def test_from_name(self):
        c = Color.from_name("red")
        assert c.r == 255
        assert c.g == 0
        assert c.b == 0

    def test_from_name_unknown(self):
        try:
            Color.from_name("notacolor")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_str_repr(self):
        c = Color(10, 20, 30)
        assert str(c) == "#0a141e"
        assert "Color" in repr(c)


class TestColorScale:
    def test_gradient(self):
        colors = ColorScale.gradient([Color(0, 0, 0), Color(255, 255, 255)], 5)
        assert len(colors) == 5
        assert isinstance(colors[0], Color)

    def test_gradient_empty(self):
        assert ColorScale.gradient([], 5) == []

    def test_gradient_single(self):
        colors = ColorScale.gradient([Color(100, 100, 100)], 3)
        assert len(colors) == 3

    def test_viridis(self):
        colors = ColorScale.viridis(10)
        assert len(colors) == 10
        assert isinstance(colors[0], Color)

    def test_plasma(self):
        colors = ColorScale.plasma(5)
        assert len(colors) == 5

    def test_grayscale(self):
        colors = ColorScale.grayscale(10)
        assert len(colors) == 10
        assert colors[0].r == 0
        assert colors[-1].r == 255

    def test_rainbow(self):
        colors = ColorScale.rainbow(10)
        assert len(colors) == 10

    def test_heat(self):
        colors = ColorScale.heat(8)
        assert len(colors) == 8

    def test_inferno(self):
        colors = ColorScale.inferno(7)
        assert len(colors) == 7

    def test_magma(self):
        colors = ColorScale.magma(6)
        assert len(colors) == 6

    def test_coolwarm(self):
        colors = ColorScale.coolwarm(9)
        assert len(colors) == 9


class TestColormap:
    def test_colormap_mid(self):
        c = colormap(50, 0, 100, "viridis")
        assert isinstance(c, Color)

    def test_colormap_edge(self):
        c = colormap(0, 0, 100, "grayscale")
        assert c.r == 0

    def test_colormap_equal_min_max(self):
        c = colormap(5, 5, 5, "viridis")
        assert isinstance(c, Color)

    def test_colormap_unknown_scale(self):
        c = colormap(50, 0, 100, "unknown")
        assert isinstance(c, Color)


class TestSVGCanvas:
    def test_rect(self):
        canvas = SVGCanvas(100, 100)
        canvas.rect(10, 10, 50, 30, fill="blue")
        result = canvas.render()
        assert "<rect" in result
        assert "blue" in result

    def test_circle(self):
        canvas = SVGCanvas(100, 100)
        canvas.circle(50, 50, 20, fill="red")
        result = canvas.render()
        assert "<circle" in result
        assert "red" in result

    def test_ellipse(self):
        canvas = SVGCanvas(100, 100)
        canvas.ellipse(50, 50, 30, 20, fill="green")
        result = canvas.render()
        assert "<ellipse" in result

    def test_line(self):
        canvas = SVGCanvas(100, 100)
        canvas.line(0, 0, 100, 100, stroke="black")
        result = canvas.render()
        assert "<line" in result

    def test_polyline(self):
        canvas = SVGCanvas(100, 100)
        canvas.polyline([(0, 0), (50, 50), (100, 0)], stroke="blue")
        result = canvas.render()
        assert "<polyline" in result

    def test_polygon(self):
        canvas = SVGCanvas(100, 100)
        canvas.polygon([(10, 10), (50, 10), (30, 50)], fill="yellow")
        result = canvas.render()
        assert "<polygon" in result

    def test_text(self):
        canvas = SVGCanvas(100, 100)
        canvas.text(50, 50, "Hello", font_size=14)
        result = canvas.render()
        assert "<text" in result
        assert "Hello" in result

    def test_path(self):
        canvas = SVGCanvas(100, 100)
        canvas.path("M 0 0 L 100 100 Z", fill="red")
        result = canvas.render()
        assert "<path" in result

    def test_render_valid_xml(self):
        canvas = SVGCanvas(200, 200)
        canvas.rect(0, 0, 200, 200, fill="white")
        canvas.circle(100, 100, 50, fill="blue")
        result = canvas.render()
        assert result.startswith("<svg")
        assert result.endswith("</svg>")
        assert 'xmlns="http://www.w3.org/2000/svg"' in result

    def test_save(self):
        canvas = SVGCanvas(100, 100)
        canvas.rect(0, 0, 100, 100, fill="red")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".svg", delete=False) as f:
            fname = f.name
        try:
            canvas.save(fname)
            with open(fname) as f:
                content = f.read()
            assert "<svg" in content
        finally:
            os.unlink(fname)


class TestBarChartSVG:
    def test_render_svg(self):
        chart = BarChartSVG({"A": 10, "B": 20, "C": 30}, title="SVG Bar")
        svg = chart.to_svg()
        result = svg.render()
        assert "<svg" in result
        assert "rect" in result


class TestLineChartSVG:
    def test_render_svg(self):
        chart = LineChartSVG(data=[1, 3, 2, 5], title="SVG Line")
        svg = chart.to_svg()
        result = svg.render()
        assert "<svg" in result

    def test_add_series(self):
        chart = LineChartSVG(data=[1, 2, 3])
        chart.add_series([3, 2, 1])
        svg = chart.to_svg()
        result = svg.render()
        assert "<svg" in result


class TestPieChartSVG:
    def test_render_svg(self):
        chart = PieChartSVG({"A": 30, "B": 70}, title="SVG Pie")
        svg = chart.to_svg()
        result = svg.render()
        assert "<svg" in result
        assert "<path" in result


class TestScatterPlotSVG:
    def test_render_svg(self):
        chart = ScatterPlotSVG(x=[1, 2, 3], y=[4, 5, 6], title="SVG Scatter")
        svg = chart.to_svg()
        result = svg.render()
        assert "<svg" in result


class TestHistogramSVG:
    def test_render_svg(self):
        chart = HistogramSVG(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bins=5, title="SVG Hist")
        svg = chart.to_svg()
        result = svg.render()
        assert "<svg" in result


class TestRadarChartSVG:
    def test_render_svg(self):
        chart = RadarChartSVG({"A": 0.5, "B": 0.8, "C": 0.3}, title="SVG Radar")
        svg = chart.to_svg()
        result = svg.render()
        assert "<svg" in result


class TestDashboardSVG:
    def test_render(self):
        dashboard = DashboardSVG(800, 600)
        dashboard.add_chart(BarChartSVG({"X": 1, "Y": 2}), 0, 0)
        dashboard.add_chart(LineChartSVG(data=[1, 2, 3]), 0, 1)
        result = dashboard.render()
        assert "<svg" in result


class TestChartJS:
    def test_bar(self):
        result = ChartJS.bar([10, 20, 30], ["A", "B", "C"], title="ChartJS Bar")
        assert "<canvas" in result
        assert "chart.js" in result
        assert "new Chart" in result

    def test_line(self):
        result = ChartJS.line([1, 2, 3], ["x", "y", "z"], title="ChartJS Line")
        assert "<canvas" in result
        assert "line" in result

    def test_line_multi(self):
        result = ChartJS.line([[1, 2, 3], [3, 2, 1]], ["a", "b", "c"],
                               datasets_labels=["S1", "S2"], title="Multi")
        assert "<canvas" in result

    def test_pie(self):
        result = ChartJS.pie([30, 70], ["A", "B"], title="ChartJS Pie")
        assert "<canvas" in result
        assert "pie" in result

    def test_scatter(self):
        result = ChartJS.scatter([1, 2, 3], [4, 5, 6], title="ChartJS Scatter")
        assert "<canvas" in result
        assert "scatter" in result

    def test_radar(self):
        result = ChartJS.radar(["A", "B", "C"], [0.5, 0.8, 0.3], title="ChartJS Radar")
        assert "<canvas" in result
        assert "radar" in result

    def test_doughnut(self):
        result = ChartJS.doughnut([10, 20, 30], ["X", "Y", "Z"], title="ChartJS Doughnut")
        assert "<canvas" in result
        assert "doughnut" in result

    def test_bubble(self):
        result = ChartJS.bubble([{"x": 1, "y": 2, "r": 5}], title="ChartJS Bubble")
        assert "<canvas" in result
        assert "bubble" in result

    def test_mixed(self):
        result = ChartJS.mixed(
            ["bar", "line"], [[1, 2, 3], [3, 2, 1]], ["a", "b", "c"], title="Mixed"
        )
        assert "<canvas" in result


class TestDashboard:
    def test_render(self):
        dash = Dashboard()
        dash.set_title("My Dashboard")
        dash.add_chart("bar", {"data": [1, 2], "labels": ["A", "B"]})
        dash.add_chart("pie", {"data": [10, 20], "labels": ["X", "Y"]})
        result = dash.render()
        assert "<!DOCTYPE html>" in result
        assert "My Dashboard" in result
        assert "chart-card" in result

    def test_save(self):
        dash = Dashboard()
        dash.add_chart("line", {"data": [1, 2], "labels": ["a", "b"], "title": "L"})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            fname = f.name
        try:
            dash.save(fname)
            with open(fname) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(fname)


class TestTableRenderer:
    def test_render(self):
        result = TableRenderer.render(
            data=[["Alice", 90], ["Bob", 85]],
            headers=["Name", "Score"],
            title="Results",
        )
        assert "Alice" in result
        assert "Bob" in result
        assert "Results" in result

    def test_striped(self):
        result = TableRenderer.render([[1]], ["A"], striped=True, bordered=True)
        assert "striped" in result
        assert "bordered" in result


class TestReport:
    def test_to_html(self):
        report = Report("Test Report")
        report.add_text("Hello World")
        report.add_section("Section 1", "Content here", level=2)
        report.add_table(["Col1", "Col2"], [["r1c1", "r1c2"], ["r2c1", "r2c2"]])
        result = report.to_html()
        assert "<!DOCTYPE html>" in result
        assert "Test Report" in result
        assert "Hello World" in result
        assert "Section 1" in result

    def test_to_markdown(self):
        report = Report("MD Report")
        report.add_text("Some text")
        report.add_section("Section A", "Detail", level=2)
        result = report.to_markdown()
        assert "# MD Report" in result
        assert "Some text" in result
        assert "## Section A" in result

    def test_save_html(self):
        report = Report("Save Test")
        report.add_text("Content")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            fname = f.name
        try:
            report.save(fname, format="html")
            with open(fname) as f:
                content = f.read()
            assert "Save Test" in content
        finally:
            os.unlink(fname)

    def test_save_markdown(self):
        report = Report("Save MD")
        report.add_text("MD content")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            fname = f.name
        try:
            report.save(fname, format="markdown")
            with open(fname) as f:
                content = f.read()
            assert "# Save MD" in content
        finally:
            os.unlink(fname)

    def test_page_break(self):
        report = Report("PB Test")
        report.add_text("Before")
        report.add_page_break()
        report.add_text("After")
        result = report.to_html()
        assert "page-break" in result

    def test_add_chart(self):
        report = Report("Chart Report")
        bar = BarChart({"A": 10, "B": 20})
        report.add_chart(bar)
        result = report.to_html()
        assert "Chart Report" in result

    def test_add_chart_svg(self):
        report = Report("SVG Chart Report")
        svg_chart = BarChartSVG({"X": 5, "Y": 10})
        report.add_chart(svg_chart)
        result = report.to_html()
        assert "SVG Chart Report" in result

    def test_custom_theme(self):
        theme = ReportTheme(primary_color="#ff0000", font_family="serif")
        report = Report("Themed Report", theme=theme)
        report.add_text("Themed content")
        result = report.to_html()
        assert "#ff0000" in result
        assert "serif" in result


class TestProgressBar:
    def test_render(self):
        bar = ProgressBar(100, description="Loading")
        bar.update(50)
        result = bar.render()
        assert "Loading" in result
        assert "50%" in result
        assert "50/100" in result

    def test_finish(self):
        bar = ProgressBar(10)
        bar.finish()
        result = bar.render()
        assert "100%" in result

    def test_no_description(self):
        bar = ProgressBar(10)
        bar.update(3)
        result = bar.render()
        assert "30%" in result


class TestLiveChart:
    def test_render(self):
        chart = LiveChart(max_points=10, width=20)
        for i in range(15):
            chart.add_point(math.sin(i * 0.5) * 10)
        result = chart.render()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty(self):
        chart = LiveChart()
        result = chart.render()
        assert isinstance(result, str)


class TestSpinner:
    def test_next_frame(self):
        spinner = Spinner()
        frames_seen = set()
        for _ in range(4):
            frames_seen.add(spinner.next_frame())
        assert len(frames_seen) > 0

    def test_custom_frames(self):
        spinner = Spinner(frames=["a", "b"])
        assert spinner.next_frame() == "a"
        assert spinner.next_frame() == "b"
        assert spinner.next_frame() == "a"


class TestTablePrinter:
    def test_render(self):
        tp = TablePrinter(headers=["Name", "Age"], widths=[15, 10])
        tp.add_row(["Alice", 30])
        tp.add_row(["Bob", 25])
        result = tp.render()
        assert "Alice" in result
        assert "Bob" in result
        assert "Name" in result
        assert "┌" in result
        assert "┘" in result

    def test_sort_by(self):
        tp = TablePrinter(headers=["X", "Y"], widths=[10, 10])
        tp.add_row(["B", 2])
        tp.add_row(["A", 1])
        tp.add_row(["C", 3])
        tp.sort_by(1)
        result = tp.render()
        lines = result.strip().split("\n")
        data_lines = [l for l in lines if "A" in l or "B" in l or "C" in l]
        assert len(data_lines) == 3

    def test_filter_rows(self):
        tp = TablePrinter(headers=["Val"], widths=[10])
        tp.add_row([10])
        tp.add_row([20])
        tp.add_row([5])
        tp.filter_rows(lambda r: r[0] >= 10)
        result = tp.render()
        assert "10" in result
        assert "20" in result
        assert "5" not in result

    def test_auto_widths(self):
        tp = TablePrinter(headers=["LongHeader", "Hi"])
        tp.add_row(["a", "b"])
        result = tp.render()
        assert "LongHeader" in result


def run_all_tests():
    test_classes = [
        TestBarChart, TestLineChart, TestPieChart, TestHistogram,
        TestScatterPlot, TestSparkline, TestHeatmap, TestBoxPlot,
        TestWaterfallChart, TestGanttChart, TestRadarChart,
        TestColor, TestColorScale, TestColormap,
        TestSVGCanvas, TestBarChartSVG, TestLineChartSVG, TestPieChartSVG,
        TestScatterPlotSVG, TestHistogramSVG, TestRadarChartSVG, TestDashboardSVG,
        TestChartJS, TestDashboard, TestTableRenderer,
        TestReport, TestProgressBar, TestLiveChart, TestSpinner, TestTablePrinter,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                try:
                    method()
                    passed += 1
                    print(f"  PASS: {cls.__name__}.{method_name}")
                except Exception as e:
                    failed += 1
                    errors.append((f"{cls.__name__}.{method_name}", str(e)))
                    print(f"  FAIL: {cls.__name__}.{method_name} - {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
