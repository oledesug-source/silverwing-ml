"""Report generation for Silverwing-ML visualization."""

import html as _html
from dataclasses import dataclass


@dataclass
class ReportTheme:
    primary_color: str = "#4e79a7"
    secondary_color: str = "#f28e2b"
    font_family: str = "sans-serif"
    background_color: str = "#ffffff"
    text_color: str = "#333333"
    heading_color: str = "#2c3e50"
    border_color: str = "#dddddd"
    width: str = "900px"
    padding: str = "20px"


@dataclass
class ReportSection:
    title: str = ""
    content: str = ""
    level: int = 1
    section_type: str = "text"


class Report:
    def __init__(self, title: str = "", theme: ReportTheme = None):
        self.title = title
        self.theme = theme or ReportTheme()
        self._sections = []

    def add_section(self, title: str, content: str, level: int = 1):
        self._sections.append(ReportSection(title=title, content=content, level=level, section_type="text"))

    def add_chart(self, chart_instance):
        html_content = ""
        if hasattr(chart_instance, "render"):
            result = chart_instance.render()
            if isinstance(result, str):
                html_content = result
        if hasattr(chart_instance, "to_svg"):
            html_content = chart_instance.to_svg().render()
        if hasattr(chart_instance, "to_html"):
            html_content = chart_instance.to_html()
        self._sections.append(ReportSection(content=html_content, section_type="chart"))

    def add_table(self, headers: list, rows: list):
        thead = "<tr>" + "".join(f"<th>{_html.escape(str(h))}</th>" for h in headers) + "</tr>"
        tbody = ""
        for row in rows:
            tds = "".join(f"<td>{_html.escape(str(cell))}</td>" for cell in row)
            tbody += f"<tr>{tds}</tr>\n"
        table_html = (
            f'<table style="border-collapse:collapse;width:100%;margin:10px 0;">\n'
            f"<thead>{thead}</thead>\n<tbody>\n{tbody}</tbody>\n</table>"
        )
        self._sections.append(ReportSection(content=table_html, section_type="table"))

    def add_text(self, content: str):
        self._sections.append(ReportSection(content=content, section_type="text"))

    def add_page_break(self):
        self._sections.append(ReportSection(content="", section_type="page_break"))

    def to_html(self) -> str:
        t = self.theme
        parts = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html lang='en'>")
        parts.append("<head>")
        parts.append("<meta charset='UTF-8'>")
        parts.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        parts.append(f"<title>{_html.escape(self.title)}</title>")
        parts.append("<style>")
        parts.append(f"body{{font-family:{t.font_family};background:{t.background_color};"
                      f"color:{t.text_color};margin:0;padding:20px;}}")
        parts.append(f".report-container{{max-width:{t.width};margin:0 auto;padding:{t.padding};"
                      f"background:white;box-shadow:0 2px 8px rgba(0,0,0,0.1);}}")
        parts.append(f"h1{{color:{t.heading_color};border-bottom:3px solid {t.primary_color};padding-bottom:10px;}}")
        parts.append(f"h2{{color:{t.heading_color};border-bottom:1px solid {t.border_color};padding-bottom:5px;}}")
        parts.append(f"h3{{color:{t.heading_color};}}")
        parts.append(".section{margin:15px 0;padding:10px 0;}")
        parts.append(f"table th{{background:{t.primary_color};color:white;padding:8px 12px;}}")
        parts.append(f"table td{{padding:8px 12px;border-bottom:1px solid {t.border_color};}}")
        parts.append(".page-break{page-break-after:always;}")
        parts.append("</style>")
        parts.append("</head>")
        parts.append("<body>")
        parts.append("<div class='report-container'>")
        parts.append(f"<h1>{_html.escape(self.title)}</h1>")

        for section in self._sections:
            if section.section_type == "page_break":
                parts.append("<div class='page-break'></div>")
                continue
            parts.append("<div class='section'>")
            if section.title:
                tag = f"h{min(section.level + 1, 4)}"
                parts.append(f"<{tag}>{_html.escape(section.title)}</{tag}>")
            if section.section_type == "chart":
                parts.append(section.content)
            elif section.section_type == "table":
                parts.append(section.content)
            else:
                parts.append(f"<p>{section.content}</p>")
            parts.append("</div>")

        parts.append("</div>")
        parts.append("</body>")
        parts.append("</html>")
        return "\n".join(parts)

    def to_markdown(self) -> str:
        lines = []
        lines.append(f"# {self.title}")
        lines.append("")

        for section in self._sections:
            if section.section_type == "page_break":
                lines.append("---")
                lines.append("")
                continue
            if section.title:
                prefix = "#" * min(section.level + 1, 4)
                lines.append(f"{prefix} {section.title}")
                lines.append("")
            if section.section_type in ("chart", "table"):
                lines.append(section.content)
                lines.append("")
            else:
                lines.append(section.content)
                lines.append("")

        return "\n".join(lines)

    def save(self, filename: str, format: str = "html"):
        content = self.to_html() if format == "html" else self.to_markdown()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
