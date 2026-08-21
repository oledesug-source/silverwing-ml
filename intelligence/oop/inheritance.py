"""Inheritance patterns: hierarchies, multiple inheritance, MRO."""

from __future__ import annotations

import csv
import io
import json
import math
from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "Vehicle", "Car", "Truck", "Motorcycle",
    "FileProcessor", "TextProcessor", "CSVProcessor", "JSONProcessor",
    "NotificationService", "EmailNotification", "SMSNotification", "PushNotification",
    "Shape", "Circle", "Rectangle", "Triangle", "Pentagon",
    "Flyer", "Swimmer", "FlyingFish",
]


class Vehicle:
    def __init__(self, make: str, model: str, year: int) -> None:
        self.make = make
        self.model = model
        self.year = year

    def start(self) -> str:
        return f"{self.year} {self.make} {self.model} started"

    def stop(self) -> str:
        return f"{self.year} {self.make} {self.model} stopped"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(make={self.make!r}, model={self.model!r}, year={self.year})"


class Car(Vehicle):
    def __init__(self, make: str, model: str, year: int, num_doors: int = 4) -> None:
        super().__init__(make, model, year)
        self.num_doors = num_doors

    def drift(self) -> str:
        return f"{self.model} is drifting!"

    def __repr__(self) -> str:
        return f"Car(make={self.make!r}, model={self.model!r}, year={self.year}, num_doors={self.num_doors})"


class Truck(Vehicle):
    def __init__(self, make: str, model: str, year: int, payload_capacity: float = 1000.0) -> None:
        super().__init__(make, model, year)
        self.payload_capacity = payload_capacity

    def haul(self, weight: float) -> str:
        if weight > self.payload_capacity:
            return f"{self.model} cannot haul {weight}kg — exceeds capacity"
        return f"{self.model} is hauling {weight}kg"

    def __repr__(self) -> str:
        return f"Truck(make={self.make!r}, model={self.model!r}, year={self.year}, payload_capacity={self.payload_capacity})"


class Motorcycle(Vehicle):
    def __init__(self, make: str, model: str, year: int, has_sidecar: bool = False) -> None:
        super().__init__(make, model, year)
        self.has_sidecar = has_sidecar

    def wheelie(self) -> str:
        return f"{self.model} does a wheelie!"

    def __repr__(self) -> str:
        return f"Motorcycle(make={self.make!r}, model={self.model!r}, year={self.year}, has_sidecar={self.has_sidecar})"


class FileProcessor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        ...


class TextProcessor(FileProcessor):
    def process(self, data: str) -> str:
        return data.upper()

    def supported_extensions(self) -> list[str]:
        return [".txt", ".text"]


class CSVProcessor(FileProcessor):
    def process(self, data: str) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(data))
        return [dict(row) for row in reader]

    def supported_extensions(self) -> list[str]:
        return [".csv"]


class JSONProcessor(FileProcessor):
    def process(self, data: str) -> Any:
        return json.loads(data)

    def supported_extensions(self) -> list[str]:
        return [".json"]


class NotificationService(ABC):
    def send(self, recipient: str, message: str) -> str:
        formatted = self.format_message(message)
        return self._deliver(recipient, formatted)

    def format_message(self, message: str) -> str:
        return f"[{type(self).__name__}] {message}"

    @abstractmethod
    def _deliver(self, recipient: str, message: str) -> str:
        ...


class EmailNotification(NotificationService):
    def _deliver(self, recipient: str, message: str) -> str:
        return f"Email to {recipient}: {message}"


class SMSNotification(NotificationService):
    def _deliver(self, recipient: str, message: str) -> str:
        return f"SMS to {recipient}: {message}"


class PushNotification(NotificationService):
    def _deliver(self, recipient: str, message: str) -> str:
        return f"Push to {recipient}: {message}"


class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        ...

    @abstractmethod
    def perimeter(self) -> float:
        ...


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


class Triangle(Shape):
    def __init__(self, a: float, b: float, c: float) -> None:
        self.a = a
        self.b = b
        self.c = c

    def area(self) -> float:
        s = (self.a + self.b + self.c) / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

    def perimeter(self) -> float:
        return self.a + self.b + self.c


class Pentagon(Shape):
    def __init__(self, side: float) -> None:
        self.side = side

    def area(self) -> float:
        return (self.side ** 2 * math.sqrt(25 + 10 * math.sqrt(5))) / 4

    def perimeter(self) -> float:
        return 5 * self.side


class Flyer:
    def fly(self) -> str:
        return f"{type(self).__name__} is flying"


class Swimmer:
    def swim(self) -> str:
        return f"{type(self).__name__} is swimming"


class FlyingFish(Flyer, Swimmer):
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"FlyingFish(name={self.name!r})"
