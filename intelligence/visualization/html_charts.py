"""HTML/CSS/JS chart generation for Silverwing-ML visualization."""

import html as _html
import json as _json


class ChartJS:
    CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js"

    @staticmethod
    def bar(data: list, labels: list, title: str = "") -> str:
        chart_id = "chart_" + str(abs(hash(title + str(labels))) % 100000)
        config = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{"label": "Data", "data": data, "backgroundColor": "rgba(78,121,167,0.7)"}],
            },
            "options": {"plugins": {"title": {"display": bool(title), "text": title}}},
        }
        return ChartJS._wrap(chart_id, config)

    @staticmethod
    def line(data: list, labels: list, datasets_labels: list = None, title: str = "") -> str:
        chart_id = "chart_" + str(abs(hash(title + str(labels))) % 100000)
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
        if isinstance(data[0], list):
            datasets = []
            for i, series in enumerate(data):
                lbl = datasets_labels[i] if datasets_labels and i < len(datasets_labels) else f"Series {i+1}"
                datasets.append({
                    "label": lbl,
                    "data": series,
                    "borderColor": colors[i % len(colors)],
                    "fill": False,
                    "tension": 0.1,
                })
        else:
            lbl = datasets_labels[0] if datasets_labels else "Data"
            datasets = [{"label": lbl, "data": data, "borderColor": colors[0], "fill": False, "tension": 0.1}]
        config = {
            "type": "line",
            "data": {"labels": labels, "datasets": datasets},
            "options": {"plugins": {"title": {"display": bool(title), "text": title}}},
        }
        return ChartJS._wrap(chart_id, config)

    @staticmethod
    def pie(data: list, labels: list, title: str = "") -> str:
        chart_id = "chart_" + str(abs(hash(title + str(labels))) % 100000)
        config = {
            "type": "pie",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [
                        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
                    ],
                }],
            },
            "options": {"plugins": {"title": {"display": bool(title), "text": title}}},
        }
        return ChartJS._wrap(chart_id, config, width=400, height=400)

    @staticmethod
    def scatter(x: list, y: list, title: str = "") -> str:
        chart_id = "chart_" + str(abs(hash(title)) % 100000)
        points = [{"x": xi, "y": yi} for xi, yi in zip(x, y)]
        config = {
            "type": "scatter",
            "data": {"datasets": [{"label": "Data", "data": points, "backgroundColor": "#4e79a7"}]},
            "options": {"plugins": {"title": {"display": bool(title), "text": title}}},
        }
        return ChartJS._wrap(chart_id, config)

    @staticmethod
    def radar(categories: list, values: list, title: str = "") -> str:
        chart_id = "chart_" + str(abs(hash(title + str(categories))) % 100000)
        config = {
            "type": "radar",
            "data": {
                "labels": categories,
                "datasets": [{"label": "Data", "data": values, "borderColor": "#4e79a7", "backgroundColor": "rgba(78,121,167,0.2)"}],
            },
            "options": {"plugins": {"title": {"display": bool(title), "text": title}}},
        }
        return ChartJS._wrap(chart_id, config, width=400, height=400)

    @staticmethod
    def doughnut(data: list, labels: list, title: str = "") -> str:
        chart_id = "chart_" + str(abs(hash(title + str(labels))) % 100000)
        config = {
            "type": "doughnut",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [
                        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
                    ],
                }],
            },
            "options": {"plugins": {"title": {"display": bool(title), "text": title}}},
        }
        return ChartJS._wrap(chart_id, config, width=400, height=400)

    @staticmethod
    def bubble(data: list, title: str = "") -> str:
        chart_id = "chart_" + str(abs(hash(title + str(len(data)))) % 100000)
        config = {
            "type": "bubble",
            "data": {"datasets": [{"label": "Data", "data": data, "backgroundColor": "rgba(78,121,167,0.5)"}]},
            "options": {"plugins": {"title": {"display": bool(title), "text": title}}},
        }
        return ChartJS._wrap(chart_id, config)

    @staticmethod
    def mixed(types: list, data: list, labels: list, title: str = "") -> str:
        chart_id = "chart_" + str(abs(hash(title + str(labels))) % 100000)
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
        datasets = []
        for i, (t, d) in enumerate(zip(types, data)):
            ds = {
                "type": t,
                "label": f"Dataset {i+1}",
                "data": d,
                "backgroundColor": colors[i % len(colors)],
            }
            if t == "line":
                ds["borderColor"] = colors[i % len(colors)]
                ds["fill"] = False
            datasets.append(ds)
        config = {
            "data": {"labels": labels, "datasets": datasets},
            "options": {"plugins": {"title": {"display": bool(title), "text": title}}},
        }
        return ChartJS._wrap(chart_id, config)

    @staticmethod
    def _wrap(chart_id: str, config: dict, width: int = 600, height: int = 400) -> str:
        config_json = _json.dumps(config, ensure_ascii=False)
        return (
            f'<canvas id="{chart_id}" width="{width}" height="{height}"></canvas>\n'
            f'<script src="{ChartJS.CHART_JS_CDN}"></script>\n'
            f"<script>\n"
            f"new Chart(document.getElementById('{chart_id}'), {config_json});\n"
            f"</script>"
        )


class Dashboard:
    def __init__(self):
        self._charts = []
        self._title = "Dashboard"

    def set_title(self, title: str):
        self._title = title

    def add_chart(self, chart_type: str, data, options: dict = None):
        self._charts.append({"type": chart_type, "data": data, "options": options or {}})

    def render(self) -> str:
        chart_divs = []
        for _i, entry in enumerate(self._charts):
            ct = entry["type"]
            d = entry["data"]
            html_content = ""
            if ct == "bar":
                html_content = ChartJS.bar(d.get("data", []), d.get("labels", []), d.get("title", ""))
            elif ct == "line":
                html_content = ChartJS.line(
                    d.get("data", []), d.get("labels", []),
                    d.get("datasets_labels"), d.get("title", ""),
                )
            elif ct == "pie":
                html_content = ChartJS.pie(d.get("data", []), d.get("labels", []), d.get("title", ""))
            elif ct == "scatter":
                html_content = ChartJS.scatter(d.get("x", []), d.get("y", []), d.get("title", ""))
            elif ct == "radar":
                html_content = ChartJS.radar(d.get("categories", []), d.get("values", []), d.get("title", ""))
            elif ct == "doughnut":
                html_content = ChartJS.doughnut(d.get("data", []), d.get("labels", []), d.get("title", ""))
            else:
                html_content = f"<p>Unsupported chart type: {_html.escape(ct)}</p>"

            chart_divs.append(
                f'<div class="chart-card">\n{html_content}\n</div>'
            )

        charts_html = "\n".join(chart_divs)
        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            "<meta charset='UTF-8'>\n"
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
            f"<title>{_html.escape(self._title)}</title>\n"
            "<style>\n"
            "body{font-family:sans-serif;margin:20px;background:#f0f2f5;}\n"
            f"h1{{text-align:center;color:#333;}}\n"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px;}\n"
            ".chart-card{background:white;border-radius:8px;padding:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}\n"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_html.escape(self._title)}</h1>\n"
            f"<div class='grid'>\n{charts_html}\n</div>\n"
            "</body>\n</html>"
        )

    def save(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.render())


class TableRenderer:
    @staticmethod
    def render(data: list, headers: list, title: str = "", striped: bool = True, bordered: bool = True) -> str:
        classes = "table"
        if striped:
            classes += " striped"
        if bordered:
            classes += " bordered"

        thead = "<tr>" + "".join(f"<th>{_html.escape(str(h))}</th>" for h in headers) + "</tr>"
        rows_html = []
        for ri, row in enumerate(data):
            stripe = "even" if ri % 2 == 0 else "odd"
            tds = "".join(f"<td>{_html.escape(str(cell))}</td>" for cell in row)
            rows_html.append(f'<tr class="{stripe}">{tds}</tr>')
        tbody = "\n".join(rows_html)

        title_html = f"<h2>{_html.escape(title)}</h2>\n" if title else ""

        return (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            "<style>\n"
            f".table{{border-collapse:collapse;width:100%;font-family:sans-serif;}}\n"
            f".table th,.table td{{padding:10px 14px;text-align:left;}}\n"
            f".table th{{background:#4e79a7;color:white;}}\n"
            f".striped .even{{background:#f9f9f9;}}\n"
            f".striped .odd{{background:#ffffff;}}\n"
            f".bordered th,.bordered td{{border:1px solid #ddd;}}\n"
            "</style>\n</head>\n<body>\n"
            f"{title_html}"
            f'<table class="{classes}">\n<thead>\n{thead}\n</thead>\n<tbody>\n{tbody}\n</tbody>\n</table>\n'
            "</body>\n</html>"
        )
