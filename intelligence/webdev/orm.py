"""
Simple ORM backed by sqlite3 with model classes, columns, and query builder.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Model",
    "Column",
    "IntegerField",
    "StringField",
    "FloatField",
    "BooleanField",
    "DateTimeField",
    "ForeignKey",
    "QueryBuilder",
]

_connection: sqlite3.Connection | None = None


def get_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """Get or create the global sqlite3 connection."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(db_path)
        _connection.row_factory = sqlite3.Row
    return _connection


def set_connection(db_path: str) -> sqlite3.Connection:
    """Set the global sqlite3 connection to a specific database file."""
    global _connection
    _connection = sqlite3.connect(db_path)
    _connection.row_factory = sqlite3.Row
    return _connection


def close_connection() -> None:
    """Close and clear the global database connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


class Column:
    """Database column definition with type and constraints."""

    def __init__(
        self,
        name: str,
        column_type: str = "TEXT",
        primary_key: bool = False,
        nullable: bool = True,
        default: Any = None,
        unique: bool = False,
        references: str | None = None,
    ) -> None:
        self.name = name
        self.column_type = column_type
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.unique = unique
        self.references = references

    def to_sql(self) -> str:
        """Generate the SQL column definition."""
        parts = [self.name, self.column_type]
        if self.primary_key:
            parts.append("PRIMARY KEY")
            parts.append("AUTOINCREMENT")
        if not self.nullable and not self.primary_key:
            parts.append("NOT NULL")
        if self.unique and not self.primary_key:
            parts.append("UNIQUE")
        if self.default is not None and not self.primary_key:
            if isinstance(self.default, str):
                parts.append(f"DEFAULT '{self.default}'")
            else:
                parts.append(f"DEFAULT {self.default}")
        if self.references:
            parts.append(f"REFERENCES {self.references}")
        return " ".join(parts)


class IntegerField(Column):
    """Integer column definition."""

    def __init__(self, name: str, primary_key: bool = False, nullable: bool = True, default: Any = None, unique: bool = False) -> None:
        super().__init__(name, "INTEGER", primary_key, nullable, default, unique)


class StringField(Column):
    """String/text column definition with optional max length."""

    def __init__(self, name: str, max_length: int = 255, nullable: bool = True, default: Any = None, unique: bool = False) -> None:
        col_type = f"VARCHAR({max_length})" if max_length <= 255 else "TEXT"
        super().__init__(name, col_type, False, nullable, default, unique)


class FloatField(Column):
    """Floating-point number column definition."""

    def __init__(self, name: str, nullable: bool = True, default: Any = None) -> None:
        super().__init__(name, "REAL", False, nullable, default)


class BooleanField(Column):
    """Boolean column definition stored as integer."""

    def __init__(self, name: str, nullable: bool = True, default: Any = None) -> None:
        super().__init__(name, "INTEGER", False, nullable, default)


class DateTimeField(Column):
    """Date/time column definition stored as text."""

    def __init__(self, name: str, nullable: bool = True, default: Any = None) -> None:
        super().__init__(name, "TEXT", False, nullable, default or "CURRENT_TIMESTAMP")


class ForeignKey(Column):
    """Foreign key column referencing another table's primary key."""

    def __init__(self, name: str, references_table: str, nullable: bool = True) -> None:
        super().__init__(name, "INTEGER", False, nullable, None, False, f"{references_table}(id)")


@dataclass
class QueryBuilder:
    """Fluent query builder for ORM-style database operations."""

    _model: type
    _conditions: list[tuple[str, str, Any]] = field(default_factory=list)
    _order_by_clause: str | None = None
    _limit_value: int | None = None
    _offset_value: int | None = None
    _select_columns: list[str] | None = None

    def where(self, **kwargs: Any) -> QueryBuilder:
        """Add WHERE conditions."""
        for key, value in kwargs.items():
            self._conditions.append(("=", key, value))
        return self

    def where_not(self, **kwargs: Any) -> QueryBuilder:
        """Add WHERE NOT conditions."""
        for key, value in kwargs.items():
            self._conditions.append(("!=", key, value))
        return self

    def where_gt(self, **kwargs: Any) -> QueryBuilder:
        """Add WHERE greater-than conditions."""
        for key, value in kwargs.items():
            self._conditions.append((">", key, value))
        return self

    def where_lt(self, **kwargs: Any) -> QueryBuilder:
        """Add WHERE less-than conditions."""
        for key, value in kwargs.items():
            self._conditions.append(("<", key, value))
        return self

    def where_in(self, field_name: str, values: list[Any]) -> QueryBuilder:
        """Add a WHERE IN condition."""
        ", ".join("?" for _ in values)
        self._conditions.append(("IN", field_name, values))
        return self

    def order_by(self, field_name: str, ascending: bool = True) -> QueryBuilder:
        """Set the ORDER BY clause."""
        direction = "ASC" if ascending else "DESC"
        self._order_by_clause = f"{field_name} {direction}"
        return self

    def limit(self, n: int) -> QueryBuilder:
        """Set the LIMIT clause."""
        self._limit_value = n
        return self

    def offset(self, n: int) -> QueryBuilder:
        """Set the OFFSET clause."""
        self._offset_value = n
        return self

    def select(self, *columns: str) -> QueryBuilder:
        """Specify columns to select."""
        self._select_columns = list(columns) if columns else None
        return self

    def execute(self) -> list[Any]:
        """Execute the query and return a list of model instances."""
        table = self._model.__table__
        cols = self._select_columns or [c.name for c in self._model.__columns__]
        columns_str = ", ".join(cols)
        query = f"SELECT {columns_str} FROM {table}"
        params: list[Any] = []

        if self._conditions:
            where_parts = []
            for op, field_name, value in self._conditions:
                if op == "IN":
                    placeholders = ", ".join("?" for _ in value)
                    where_parts.append(f"{field_name} IN ({placeholders})")
                    params.extend(value)
                else:
                    where_parts.append(f"{field_name} {op} ?")
                    params.append(value)
            query += " WHERE " + " AND ".join(where_parts)

        if self._order_by_clause:
            query += f" ORDER BY {self._order_by_clause}"
        if self._limit_value is not None:
            query += f" LIMIT {self._limit_value}"
        if self._offset_value is not None:
            query += f" OFFSET {self._offset_value}"

        conn = get_connection()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [self._model._from_row(row) for row in rows]


class ModelMeta(type):
    """Metaclass that processes Column definitions for Model subclasses."""

    def __new__(mcs, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> Any:
        columns: list[Column] = []
        if name == "Model":
            return super().__new__(mcs, name, bases, attrs)
        for key, value in list(attrs.items()):
            if isinstance(value, Column):
                if not value.name:
                    value.name = key
                columns.append(value)
        attrs["__columns__"] = columns
        table_name = attrs.get("__table__", name.lower() + "s")
        attrs["__table__"] = table_name
        return super().__new__(mcs, name, bases, attrs)


class Model(metaclass=ModelMeta):
    """Base ORM model class backed by sqlite3."""

    __table__: str = ""
    __columns__: list[Column] = []
    id: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        for col in self.__columns__:
            if col.name in kwargs:
                setattr(self, col.name, kwargs[col.name])
            elif col.primary_key:
                setattr(self, col.name, kwargs.get("id", None))
            else:
                default = col.default() if callable(col.default) else col.default
                setattr(self, col.name, kwargs.get(col.name, default))
        if "id" in kwargs:
            self.id = kwargs["id"]

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> Model:
        """Create a model instance from a database row."""
        data = dict(row)
        return cls(**data)

    @classmethod
    def create_table(cls) -> None:
        """Create the database table for this model."""
        col_defs = ", ".join(col.to_sql() for col in cls.__columns__)
        if not any(c.primary_key for c in cls.__columns__):
            col_defs = "id INTEGER PRIMARY KEY AUTOINCREMENT, " + col_defs
        query = f"CREATE TABLE IF NOT EXISTS {cls.__table__} ({col_defs})"
        conn = get_connection()
        conn.execute(query)
        conn.commit()

    @classmethod
    def drop_table(cls) -> None:
        """Drop the database table for this model."""
        conn = get_connection()
        conn.execute(f"DROP TABLE IF EXISTS {cls.__table__}")
        conn.commit()

    def save(self) -> None:
        """Insert or update this model instance in the database."""
        conn = get_connection()
        col_names = [c.name for c in self.__columns__ if not c.primary_key or c.name == "id"]
        [getattr(self, name, None) for name in col_names]
        if self.id is not None:
            set_clause = ", ".join(f"{name} = ?" for name in col_names if name != "id")
            set_values = [getattr(self, name, None) for name in col_names if name != "id"]
            query = f"UPDATE {self.__table__} SET {set_clause} WHERE id = ?"
            conn.execute(query, set_values + [self.id])
        else:
            insert_cols = [n for n in col_names if n != "id"]
            insert_vals = [getattr(self, n, None) for n in insert_cols]
            placeholders = ", ".join("?" for _ in insert_cols)
            query = f"INSERT INTO {self.__table__} ({', '.join(insert_cols)}) VALUES ({placeholders})"
            cursor = conn.execute(query, insert_vals)
            self.id = cursor.lastrowid
        conn.commit()

    def delete(self) -> None:
        """Delete this model instance from the database."""
        if self.id is None:
            return
        conn = get_connection()
        conn.execute(f"DELETE FROM {self.__table__} WHERE id = ?", (self.id,))
        conn.commit()

    @classmethod
    def get(cls, record_id: int) -> Model | None:
        """Get a single record by primary key."""
        conn = get_connection()
        cursor = conn.execute(f"SELECT * FROM {cls.__table__} WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        return cls._from_row(row) if row else None

    @classmethod
    def filter(cls, **kwargs: Any) -> list[Model]:
        """Filter records by keyword arguments."""
        return QueryBuilder(cls).where(**kwargs).execute()

    @classmethod
    def all(cls) -> list[Model]:
        """Get all records from the table."""
        return QueryBuilder(cls).execute()

    @classmethod
    def count(cls) -> int:
        """Count all records in the table."""
        conn = get_connection()
        cursor = conn.execute(f"SELECT COUNT(*) FROM {cls.__table__}")
        return cursor.fetchone()[0]

    @classmethod
    def select(cls, *columns: str) -> QueryBuilder:
        """Start a query with specific columns."""
        builder = QueryBuilder(cls)
        if columns:
            builder = builder.select(*columns)
        return builder

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id}>"
