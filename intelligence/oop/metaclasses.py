"""Advanced OOP: metaclasses."""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Callable
from typing import Any

__all__ = [
    "AutoRegisterMeta", "ValidatedFieldsMeta", "FrozenInstanceMeta",
    "SingletonMeta", "AbstractBaseMeta",
]


_auto_registry: dict[str, type] = {}


class AutoRegisterMeta(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs: Any) -> AutoRegisterMeta:
        cls = super().__new__(mcs, name, bases, namespace)
        _auto_registry[name] = cls
        return cls

    @classmethod
    def get_registry(mcs) -> dict[str, type]:
        return dict(_auto_registry)

    @classmethod
    def get(mcs, name: str) -> type | None:
        return _auto_registry.get(name)


class ValidatedFieldsMeta(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs: Any) -> ValidatedFieldsMeta:
        annotations: dict[str, Any] = {}
        for base in reversed(bases):
            if hasattr(base, "__annotations__"):
                annotations.update(base.__annotations__)
        annotations.update(namespace.get("__annotations__", {}))

        validators: dict[str, Callable] = {}
        for _key, value in namespace.items():
            if callable(value) and hasattr(value, "_field_validator"):
                validators[value._field_validator] = value

        cls = super().__new__(mcs, name, bases, namespace)

        for field_name, _field_type in annotations.items():
            if field_name.startswith("_"):
                continue
            if field_name in validators:
                validator_fn = validators[field_name]

                def _make_getter(fn: str) -> Callable:
                    def _getter(self: Any) -> Any:
                        return getattr(self, f"_val_{fn}", None)
                    return _getter

                def _make_setter(vfn: Callable, fn: str) -> Callable:
                    def _setter(self: Any, value: Any) -> None:
                        if not vfn(value):
                            raise ValueError(f"Validation failed for '{fn}'")
                        setattr(self, f"_val_{fn}", value)
                    return _setter

                setattr(cls, field_name, property(
                    fget=_make_getter(field_name),
                    fset=_make_setter(validator_fn, field_name),
                ))

        return cls


class FrozenInstanceMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        object.__setattr__(instance, "_frozen", True)
        return instance

    def __init__(cls, name: str, bases: tuple, namespace: dict) -> None:
        super().__init__(name, bases, namespace)
        original_setattr = cls.__setattr__

        def frozen_setattr(self: Any, key: str, value: Any) -> None:
            if getattr(self, "_frozen", False):
                raise AttributeError(
                    f"Cannot set attribute on frozen instance of {type(self).__name__}"
                )
            original_setattr(self, key, value)

        cls.__setattr__ = frozen_setattr


class SingletonMeta(type):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def clear(mcs, cls: type) -> None:
        mcs._instances.pop(cls, None)


class AbstractBaseMeta(ABCMeta):
    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs: Any) -> AbstractBaseMeta:
        cls = super().__new__(mcs, name, bases, namespace)
        return cls
