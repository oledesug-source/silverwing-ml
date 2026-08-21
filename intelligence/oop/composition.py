"""Composition vs Inheritance: HAS-A relationships, mixins."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "Engine", "Car", "Address", "User",
    "OrderItem", "Order",
    "Student", "Course", "Department", "University",
    "Folder", "File", "FileSystem",
    "Observable",
]


class Engine:
    def __init__(self, horsepower: int, fuel_type: str = "gasoline") -> None:
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.running = False

    def start(self) -> str:
        self.running = True
        return f"Engine ({self.horsepower}HP {self.fuel_type}) started"

    def stop(self) -> str:
        self.running = False
        return "Engine stopped"

    def __repr__(self) -> str:
        return f"Engine(horsepower={self.horsepower}, fuel_type={self.fuel_type!r})"


class Car:
    def __init__(self, make: str, model: str, engine: Engine) -> None:
        self.make = make
        self.model = model
        self.engine = engine

    def start(self) -> str:
        return f"{self.make} {self.model}: {self.engine.start()}"

    def stop(self) -> str:
        return f"{self.make} {self.model}: {self.engine.stop()}"

    def __repr__(self) -> str:
        return f"Car(make={self.make!r}, model={self.model!r}, engine={self.engine})"


class Address:
    def __init__(self, street: str, city: str, state: str, zip_code: str) -> None:
        self.street = street
        self.city = city
        self.state = state
        self.zip_code = zip_code

    def full_address(self) -> str:
        return f"{self.street}, {self.city}, {self.state} {self.zip_code}"

    def __repr__(self) -> str:
        return f"Address(street={self.street!r}, city={self.city!r}, state={self.state!r}, zip_code={self.zip_code!r})"


class User:
    def __init__(self, name: str, email: str, address: Address) -> None:
        self.name = name
        self.email = email
        self.address = address

    def get_shipping_address(self) -> str:
        return self.address.full_address()

    def __repr__(self) -> str:
        return f"User(name={self.name!r}, email={self.email!r})"


class OrderItem:
    def __init__(self, product: str, price: float, quantity: int = 1) -> None:
        self.product = product
        self.price = price
        self.quantity = quantity

    def subtotal(self) -> float:
        return self.price * self.quantity

    def __repr__(self) -> str:
        return f"OrderItem(product={self.product!r}, price={self.price}, quantity={self.quantity})"


class Order:
    def __init__(self, customer: User, tax_rate: float = 0.08) -> None:
        self.customer = customer
        self.items: list[OrderItem] = []
        self.tax_rate = tax_rate

    def add_item(self, item: OrderItem) -> None:
        self.items.append(item)

    def compute_subtotal(self) -> float:
        return sum(item.subtotal() for item in self.items)

    def compute_tax(self) -> float:
        return self.compute_subtotal() * self.tax_rate

    def compute_total(self) -> float:
        return self.compute_subtotal() + self.compute_tax()

    def apply_discount(self, percent: float) -> float:
        discount = self.compute_subtotal() * (percent / 100)
        return self.compute_subtotal() - discount + self.compute_tax()

    def __repr__(self) -> str:
        return f"Order(customer={self.customer.name!r}, items={len(self.items)})"


class Student:
    def __init__(self, name: str, student_id: str) -> None:
        self.name = name
        self.student_id = student_id
        self.courses: list[Course] = []

    def enroll(self, course: Course) -> None:
        self.courses.append(course)
        course.students.append(self)

    def __repr__(self) -> str:
        return f"Student(name={self.name!r}, student_id={self.student_id!r})"


class Course:
    def __init__(self, name: str, credits: int = 3) -> None:
        self.name = name
        self.credits = credits
        self.students: list[Student] = []
        self.department: Department | None = None

    def add_student(self, student: Student) -> None:
        if student not in self.students:
            self.students.append(student)
            student.courses.append(self)

    def __repr__(self) -> str:
        return f"Course(name={self.name!r}, credits={self.credits})"


class Department:
    def __init__(self, name: str) -> None:
        self.name = name
        self.courses: list[Course] = []
        self.university: University | None = None

    def add_course(self, course: Course) -> None:
        self.courses.append(course)
        course.department = self

    def __repr__(self) -> str:
        return f"Department(name={self.name!r})"


class University:
    def __init__(self, name: str) -> None:
        self.name = name
        self.departments: list[Department] = []

    def add_department(self, department: Department) -> None:
        self.departments.append(department)
        department.university = self

    def total_students(self) -> int:
        count = 0
        for dept in self.departments:
            for course in dept.courses:
                count += len(course.students)
        return count

    def __repr__(self) -> str:
        return f"University(name={self.name!r})"


class FileNode:
    def __init__(self, name: str) -> None:
        self.name = name


class File(FileNode):
    def __init__(self, name: str, size: int = 0) -> None:
        super().__init__(name)
        self.size = size

    def __repr__(self) -> str:
        return f"File(name={self.name!r}, size={self.size})"


class Folder(FileNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.children: list[FileNode] = []

    def add(self, item: FileNode) -> None:
        self.children.append(item)

    def ls(self) -> list[str]:
        return [child.name for child in self.children]

    def find(self, name: str) -> FileNode | None:
        for child in self.children:
            if child.name == name:
                return child
            if isinstance(child, Folder):
                found = child.find(name)
                if found is not None:
                    return found
        return None

    def du(self) -> int:
        total = 0
        for child in self.children:
            if isinstance(child, File):
                total += child.size
            elif isinstance(child, Folder):
                total += child.du()
        return total

    def __repr__(self) -> str:
        return f"Folder(name={self.name!r})"


class FileSystem:
    def __init__(self) -> None:
        self.root = Folder("/")

    def ls(self, path: str = "/") -> list[str]:
        node = self._navigate(path)
        if isinstance(node, Folder):
            return node.ls()
        return []

    def find(self, name: str) -> FileNode | None:
        return self.root.find(name)

    def du(self, path: str = "/") -> int:
        node = self._navigate(path)
        if isinstance(node, Folder):
            return node.du()
        return 0

    def _navigate(self, path: str) -> FileNode:
        if path == "/":
            return self.root
        parts = path.strip("/").split("/")
        current: FileNode = self.root
        for part in parts:
            if isinstance(current, Folder):
                found = current.find(part)
                if found is None:
                    raise FileNotFoundError(f"Path not found: {path}")
                current = found
            else:
                raise FileNotFoundError(f"Path not found: {path}")
        return current

    def __repr__(self) -> str:
        return "FileSystem()"


class Observable:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = {}

    def subscribe(self, event: str, callback: Callable) -> None:
        self._subscribers.setdefault(event, []).append(callback)

    def unsubscribe(self, event: str, callback: Callable) -> None:
        if event in self._subscribers:
            self._subscribers[event] = [cb for cb in self._subscribers[event] if cb != callback]

    def notify(self, event: str, data: Any = None) -> list[Any]:
        results = []
        for callback in self._subscribers.get(event, []):
            results.append(callback(data))
        return results
