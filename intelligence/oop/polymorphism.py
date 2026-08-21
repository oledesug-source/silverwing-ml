"""Polymorphism and duck typing."""

from __future__ import annotations

import csv
import io
import json
import math
import xml.etree.ElementTree as ET
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "draw", "process", "Serializer", "JSONSerializer", "CSVSerializer",
    "XMLSerializer", "Calculator", "Polynomial",
]


def draw(shape: Any) -> str:
    area = shape.area()
    return f"Drawing shape with area {area:.2f}"


def process(items: Any) -> int:
    return len(items)


@runtime_checkable
class Serializer(Protocol):
    def serialize(self, obj: Any) -> str:
        ...


class JSONSerializer:
    def serialize(self, obj: Any) -> str:
        return json.dumps(obj)


class CSVSerializer:
    def serialize(self, obj: Any) -> str:
        if not obj:
            return ""
        if isinstance(obj, list) and isinstance(obj[0], dict):
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=obj[0].keys())
            writer.writeheader()
            writer.writerows(obj)
            return buf.getvalue().strip()
        return json.dumps(obj)


class XMLSerializer:
    def serialize(self, obj: Any) -> str:
        root = ET.Element("data")
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = ET.SubElement(root, str(key))
                child.text = str(value)
        elif isinstance(obj, list):
            for item in obj:
                child = ET.SubElement(root, "item")
                child.text = str(item)
        else:
            root.text = str(obj)
        return ET.tostring(root, encoding="unicode")


class Calculator:
    def __init__(self, value: float = 0) -> None:
        self.value = value

    def __add__(self, other: Calculator | float) -> Calculator:
        if isinstance(other, Calculator):
            return Calculator(self.value + other.value)
        return Calculator(self.value + other)

    def __sub__(self, other: Calculator | float) -> Calculator:
        if isinstance(other, Calculator):
            return Calculator(self.value - other.value)
        return Calculator(self.value - other)

    def __mul__(self, other: Calculator | float) -> Calculator:
        if isinstance(other, Calculator):
            return Calculator(self.value * other.value)
        return Calculator(self.value * other)

    def __truediv__(self, other: Calculator | float) -> Calculator:
        if isinstance(other, Calculator):
            return Calculator(self.value / other.value)
        return Calculator(self.value / other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Calculator):
            return math.isclose(self.value, other.value)
        if isinstance(other, (int, float)):
            return math.isclose(self.value, other)
        return NotImplemented

    def __lt__(self, other: Calculator | float) -> bool:
        if isinstance(other, Calculator):
            return self.value < other.value
        return self.value < other

    def __repr__(self) -> str:
        return f"Calculator({self.value})"


class Polynomial:
    def __init__(self, coefficients: list[float]) -> None:
        self.coefficients = list(coefficients)

    def evaluate(self, x: float) -> float:
        return sum(c * x ** i for i, c in enumerate(self.coefficients))

    def __call__(self, x: float) -> float:
        return self.evaluate(x)

    def __add__(self, other: Polynomial) -> Polynomial:
        max_len = max(len(self.coefficients), len(other.coefficients))
        a = self.coefficients + [0] * (max_len - len(self.coefficients))
        b = other.coefficients + [0] * (max_len - len(other.coefficients))
        return Polynomial([x + y for x, y in zip(a, b)])

    def __mul__(self, other: Polynomial) -> Polynomial:
        result = [0.0] * (len(self.coefficients) + len(other.coefficients) - 1)
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                result[i + j] += a * b
        return Polynomial(result)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.coefficients == other.coefficients

    def __repr__(self) -> str:
        return f"Polynomial({self.coefficients})"
