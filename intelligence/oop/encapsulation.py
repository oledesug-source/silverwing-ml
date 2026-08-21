"""Encapsulation: private attributes, properties, descriptors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "Person", "BankAccount", "Product", "Temperature",
    "ReadOnlyAttribute", "ValidatedAttribute",
]


class Person:
    def __init__(self, name: str, age: int) -> None:
        self._name = name
        self._age = age

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value

    @property
    def age(self) -> int:
        return self._age

    @age.setter
    def age(self, value: int) -> None:
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value

    def __repr__(self) -> str:
        return f"Person(name={self._name!r}, age={self._age})"


class BankAccount:
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

    def __repr__(self) -> str:
        return f"BankAccount(owner={self.owner!r}, balance={self._balance})"


class Product:
    def __init__(self, name: str, price: float) -> None:
        self._name = name
        self._price = price

    @property
    def name(self) -> str:
        return self._name

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, value: float) -> None:
        if value < 0:
            raise ValueError("Price cannot be negative")
        self._price = value

    def discount(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("Discount amount cannot be negative")
        self._price = max(0.0, self._price - amount)

    def apply_tax(self, rate: float) -> None:
        if rate < 0:
            raise ValueError("Tax rate cannot be negative")
        self._price *= (1 + rate)

    def __repr__(self) -> str:
        return f"Product(name={self._name!r}, price={self._price})"


class Temperature:
    def __init__(self, celsius: float | None = None, fahrenheit: float | None = None) -> None:
        if celsius is not None and fahrenheit is not None:
            raise ValueError("Provide only celsius or fahrenheit, not both")
        if celsius is not None:
            self._celsius = celsius
        elif fahrenheit is not None:
            self._celsius = (fahrenheit - 32) * 5 / 9
        else:
            raise ValueError("Provide celsius or fahrenheit")

    @property
    def celsius(self) -> float:
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        self._celsius = (value - 32) * 5 / 9

    def __repr__(self) -> str:
        return f"Temperature(celsius={self._celsius})"


class ReadOnlyAttribute:
    def __init__(self, default: Any = None) -> None:
        self.default = default

    def set_name(self, owner: type, name: str) -> None:
        self.name = name

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type = None) -> Any:
        if obj is None:
            return self
        return getattr(obj, f"_ro_{self.name}", self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        if hasattr(obj, f"_ro_{self.name}"):
            raise AttributeError(f"{self.name} is read-only after initialization")
        setattr(obj, f"_ro_{self.name}", value)


class ValidatedAttribute:
    def __init__(self, validator: Callable[[Any], bool], error_msg: str = "Validation failed") -> None:
        self.validator = validator
        self.error_msg = error_msg

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type = None) -> Any:
        if obj is None:
            return self
        return getattr(obj, f"_va_{self.name}", None)

    def __set__(self, obj: Any, value: Any) -> None:
        if not self.validator(value):
            raise ValueError(self.error_msg)
        setattr(obj, f"_va_{self.name}", value)
