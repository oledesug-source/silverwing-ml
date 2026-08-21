"""Design patterns implemented in pure Python."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

__all__ = [
    "SingletonMeta", "Database", "Logger",
    "ShapeFactory",
    "EventEmitter",
    "Sorter", "QuickSort", "MergeSort", "HeapSort",
    "Coffee", "Espresso", "Decorator", "MilkDecorator", "SugarDecorator", "WhipDecorator",
    "Command", "MacroCommand", "History",
    "VendingMachine",
    "QueryBuilder",
    "PaymentProcessor", "LegacyPaymentProcessor",
    "ImageProxy", "AccessControlProxy",
]


class SingletonMeta(type):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Database(metaclass=SingletonMeta):
    def __init__(self, connection_string: str = "") -> None:
        self.connection_string = connection_string
        self.connected = True

    def query(self, sql: str) -> str:
        return f"Executing: {sql}"

    def __repr__(self) -> str:
        return f"Database(connection_string={self.connection_string!r})"


class Logger(metaclass=SingletonMeta):
    def __init__(self, name: str = "app") -> None:
        self.name = name
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        self.logs.append(message)

    def get_logs(self) -> list[str]:
        return list(self.logs)

    def __repr__(self) -> str:
        return f"Logger(name={self.name!r})"


class ShapeFactory:
    @staticmethod
    def create(shape_type: str, **kwargs: Any) -> Any:
        from intelligence.oop.inheritance import Circle, Rectangle, Triangle
        shape_type = shape_type.lower()
        if shape_type == "circle":
            return Circle(kwargs.get("radius", 1))
        if shape_type == "rectangle":
            return Rectangle(kwargs.get("w", 1), kwargs.get("h", 1))
        if shape_type == "triangle":
            return Triangle(
                kwargs.get("a", 1),
                kwargs.get("b", 1),
                kwargs.get("c", 1),
            )
        raise ValueError(f"Unknown shape type: {shape_type}")


class EventEmitter:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = {}

    def on(self, event: str, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable) -> None:
        if event in self._listeners:
            self._listeners[event] = [cb for cb in self._listeners[event] if cb != callback]

    def emit(self, event: str, data: Any = None) -> list[Any]:
        results = []
        for callback in self._listeners.get(event, []):
            results.append(callback(data))
        return results


class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list[Any]) -> list[Any]:
        ...


class QuickSort(SortStrategy):
    def sort(self, data: list[Any]) -> list[Any]:
        if len(data) <= 1:
            return list(data)
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class MergeSort(SortStrategy):
    def sort(self, data: list[Any]) -> list[Any]:
        if len(data) <= 1:
            return list(data)
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return self._merge(left, right)

    def _merge(self, left: list[Any], right: list[Any]) -> list[Any]:
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result


class HeapSort(SortStrategy):
    def sort(self, data: list[Any]) -> list[Any]:
        arr = list(data)
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self._heapify(arr, i, 0)
        return arr

    def _heapify(self, arr: list[Any], n: int, i: int) -> None:
        largest = i
        l, r = 2 * i + 1, 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self._heapify(arr, n, largest)


class Sorter:
    def __init__(self, strategy: SortStrategy | None = None) -> None:
        self._strategy = strategy or QuickSort()

    def set_strategy(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def sort(self, data: list[Any]) -> list[Any]:
        return self._strategy.sort(data)


class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float:
        ...

    @abstractmethod
    def description(self) -> str:
        ...


class Espresso(Coffee):
    def cost(self) -> float:
        return 2.00

    def description(self) -> str:
        return "Espresso"


class Decorator(Coffee, ABC):
    def __init__(self, coffee: Coffee) -> None:
        self._coffee = coffee


class MilkDecorator(Decorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.50

    def description(self) -> str:
        return self._coffee.description() + " + Milk"


class SugarDecorator(Decorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.25

    def description(self) -> str:
        return self._coffee.description() + " + Sugar"


class WhipDecorator(Decorator):
    def cost(self) -> float:
        return self._coffee.cost() + 0.75

    def description(self) -> str:
        return self._coffee.description() + " + Whip"


class Command(ABC):
    @abstractmethod
    def execute(self) -> Any:
        ...

    @abstractmethod
    def undo(self) -> Any:
        ...


class _TextCommand(Command):
    def __init__(self, text_obj: _TextBuffer, text: str) -> None:
        self._buffer = text_obj
        self._text = text

    def execute(self) -> None:
        self._buffer.content += self._text

    def undo(self) -> None:
        self._buffer.content = self._buffer.content[: -len(self._text)]


class _TextBuffer:
    def __init__(self) -> None:
        self.content: str = ""


class MacroCommand(Command):
    def __init__(self, commands: list[Command]) -> None:
        self._commands = list(commands)

    def execute(self) -> list[Any]:
        return [cmd.execute() for cmd in self._commands]

    def undo(self) -> list[Any]:
        return [cmd.undo() for cmd in reversed(self._commands)]


class History:
    def __init__(self) -> None:
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    def execute(self, command: Command) -> Any:
        result = command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()
        return result

    def undo(self) -> Any:
        if not self._undo_stack:
            raise RuntimeError("Nothing to undo")
        command = self._undo_stack.pop()
        self._redo_stack.append(command)
        return command.undo()

    def redo(self) -> Any:
        if not self._redo_stack:
            raise RuntimeError("Nothing to redo")
        command = self._redo_stack.pop()
        self._undo_stack.append(command)
        return command.execute()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0


def _make_insert_command(table: str, data: dict[str, Any]) -> Command:
    class _InsertCommand(Command):
        def __init__(self, table: str, data: dict[str, Any]) -> None:
            self.table = table
            self.data = data

        def execute(self) -> str:
            return f"Inserted into {self.table}: {self.data}"

        def undo(self) -> str:
            return f"Undid insert into {self.table}"
    return _InsertCommand(table, data)


Command._make_insert = staticmethod(_make_insert_command)


class State(ABC):
    @abstractmethod
    def insert_coin(self, machine: VendingMachine) -> str:
        ...

    @abstractmethod
    def select_item(self, machine: VendingMachine) -> str:
        ...

    @abstractmethod
    def dispense(self, machine: VendingMachine) -> str:
        ...


class IdleState(State):
    def insert_coin(self, machine: VendingMachine) -> str:
        machine.set_state(machine._has_coin_state)
        return "Coin inserted"

    def select_item(self, machine: VendingMachine) -> str:
        return "Insert coin first"

    def dispense(self, machine: VendingMachine) -> str:
        return "Insert coin first"


class HasCoinState(State):
    def insert_coin(self, machine: VendingMachine) -> str:
        return "Coin already inserted"

    def select_item(self, machine: VendingMachine) -> str:
        machine.set_state(machine._dispensing_state)
        return "Item selected"

    def dispense(self, machine: VendingMachine) -> str:
        return "Select item first"


class DispensingState(State):
    def insert_coin(self, machine: VendingMachine) -> str:
        return "Please wait, dispensing"

    def select_item(self, machine: VendingMachine) -> str:
        return "Please wait, dispensing"

    def dispense(self, machine: VendingMachine) -> str:
        machine.stock -= 1
        if machine.stock <= 0:
            machine.set_state(machine._out_of_stock_state)
        else:
            machine.set_state(machine._idle_state)
        return "Item dispensed"


class OutOfStockState(State):
    def insert_coin(self, machine: VendingMachine) -> str:
        return "Machine out of stock"

    def select_item(self, machine: VendingMachine) -> str:
        return "Machine out of stock"

    def dispense(self, machine: VendingMachine) -> str:
        return "Machine out of stock"


class VendingMachine:
    def __init__(self, stock: int = 5) -> None:
        self.stock = stock
        self._idle_state = IdleState()
        self._has_coin_state = HasCoinState()
        self._dispensing_state = DispensingState()
        self._out_of_stock_state = OutOfStockState()
        self._state: State = self._idle_state

    def set_state(self, state: State) -> None:
        self._state = state

    def insert_coin(self) -> str:
        return self._state.insert_coin(self)

    def select_item(self) -> str:
        return self._state.select_item(self)

    def dispense(self) -> str:
        return self._state.dispense(self)


class QueryBuilder:
    def __init__(self) -> None:
        self._select_cols: list[str] = []
        self._from_table: str = ""
        self._where_clauses: list[str] = []
        self._order_by_col: str = ""
        self._limit_val: int | None = None

    def select(self, *columns: str) -> QueryBuilder:
        self._select_cols = list(columns)
        return self

    def from_table(self, table: str) -> QueryBuilder:
        self._from_table = table
        return self

    def where(self, condition: str) -> QueryBuilder:
        self._where_clauses.append(condition)
        return self

    def order_by(self, column: str) -> QueryBuilder:
        self._order_by_col = column
        return self

    def limit(self, n: int) -> QueryBuilder:
        self._limit_val = n
        return self

    def build(self) -> str:
        cols = ", ".join(self._select_cols) if self._select_cols else "*"
        parts = [f"SELECT {cols}", f"FROM {self._from_table}"]
        if self._where_clauses:
            parts.append("WHERE " + " AND ".join(self._where_clauses))
        if self._order_by_col:
            parts.append(f"ORDER BY {self._order_by_col}")
        if self._limit_val is not None:
            parts.append(f"LIMIT {self._limit_val}")
        return " ".join(parts)


class PaymentProcessor(ABC):
    @abstractmethod
    def pay(self, amount: float) -> str:
        ...


class LegacyPaymentProcessor:
    def process_payment(self, amount: float, currency: str) -> str:
        return f"Legacy: Paid {amount} {currency}"


class PaymentAdapter(PaymentProcessor):
    def __init__(self, legacy: LegacyPaymentProcessor) -> None:
        self._legacy = legacy

    def pay(self, amount: float) -> str:
        return self._legacy.process_payment(amount, "USD")


class Image:
    def __init__(self, path: str) -> None:
        self.path = path
        self.loaded = True

    def display(self) -> str:
        return f"Displaying {self.path}"


class ImageProxy:
    def __init__(self, path: str) -> None:
        self._path = path
        self._image: Image | None = None

    def display(self) -> str:
        if self._image is None:
            self._image = Image(self._path)
        return self._image.display()


class AccessControlProxy:
    def __init__(self, target: Any, allowed_roles: list[str]) -> None:
        self._target = target
        self._allowed_roles = allowed_roles

    def execute(self, user_role: str, method: str, *args: Any, **kwargs: Any) -> Any:
        if user_role not in self._allowed_roles:
            raise PermissionError(f"Access denied for role: {user_role}")
        return getattr(self._target, method)(*args, **kwargs)
