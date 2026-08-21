"""OOP Fundamentals: classes, inheritance, encapsulation basics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

__all__ = [
    "Animal", "Dog", "Cat", "Bird",
    "Shape", "Circle", "Rectangle", "Triangle",
    "Employee", "BankAccount", "Vector2D",
]


class Animal(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def speak(self) -> str:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name} says Woof!"


class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says Meow!"


class Bird(Animal):
    def speak(self) -> str:
        return f"{self.name} says Tweet!"


class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        ...

    @abstractmethod
    def perimeter(self) -> float:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

    def __repr__(self) -> str:
        return f"Circle(radius={self.radius})"


class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __repr__(self) -> str:
        return f"Rectangle(width={self.width}, height={self.height})"


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

    def __repr__(self) -> str:
        return f"Triangle(a={self.a}, b={self.b}, c={self.c})"


class Employee:
    def __init__(self, name: str, salary: float, department: str) -> None:
        self.name = name
        self.salary = salary
        self.department = department

    def promote(self) -> None:
        self.salary *= 1.10
        self.department = "Senior " + self.department

    def give_raise(self, amount: float) -> None:
        self.salary += amount

    def __str__(self) -> str:
        return f"{self.name} | {self.department} | ${self.salary:,.2f}"


class BankAccount:
    interest_rate: float = 0.05

    def __init__(self, owner: str, balance: float = 0.0) -> None:
        self.owner = owner
        self._balance = balance

    @property
    def balance(self) -> float:
        return self._balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

    def transfer(self, other: BankAccount, amount: float) -> None:
        self.withdraw(amount)
        other.deposit(amount)

    def apply_interest(self) -> None:
        self._balance += self._balance * self.interest_rate

    def get_statement(self) -> str:
        return f"Account: {self.owner} | Balance: ${self._balance:,.2f}"

    def __repr__(self) -> str:
        return f"BankAccount(owner={self.owner!r}, balance={self._balance})"


class Vector2D:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def add(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def subtract(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> Vector2D:
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def dot(self, other: Vector2D) -> float:
        return self.x * other.x + self.y * other.y

    def angle_between(self, other: Vector2D) -> float:
        cos_theta = self.dot(other) / (self.magnitude() * other.magnitude())
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta)

    def __add__(self, other: Vector2D) -> Vector2D:
        return self.add(other)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return self.subtract(other)

    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector2D):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"
