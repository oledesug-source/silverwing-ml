"""
Form handling, validation, and HTML rendering.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Callable
from typing import Any

__all__ = [
    "Field",
    "TextField",
    "EmailField",
    "IntegerField",
    "FloatField",
    "BooleanField",
    "DateField",
    "ChoiceField",
    "MultipleChoiceField",
    "FileField",
    "PasswordField",
    "Form",
    "ModelForm",
    "EmailValidator",
    "URLValidator",
    "MinLengthValidator",
    "MaxLengthValidator",
    "RangeValidator",
    "RegexValidator",
]


class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class EmailValidator:
    """Validates email address format."""

    def __init__(self, message: str = "Invalid email address") -> None:
        self.message = message
        self.pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    def __call__(self, value: str) -> bool:
        if not self.pattern.match(str(value)):
            raise ValidationError(self.message)
        return True


class URLValidator:
    """Validates URL format."""

    def __init__(self, message: str = "Invalid URL") -> None:
        self.message = message
        self.pattern = re.compile(
            r"^https?://"
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
            r"localhost|"
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
            r"(?::\d+)?"
            r"(?:/?|[/?]\S+)$", re.IGNORECASE
        )

    def __call__(self, value: str) -> bool:
        if not self.pattern.match(str(value)):
            raise ValidationError(self.message)
        return True


class MinLengthValidator:
    """Validates minimum string length."""

    def __init__(self, min_length: int, message: str | None = None) -> None:
        self.min_length = min_length
        self.message = message or f"Minimum length is {min_length}"

    def __call__(self, value: Any) -> bool:
        if len(str(value)) < self.min_length:
            raise ValidationError(self.message)
        return True


class MaxLengthValidator:
    """Validates maximum string length."""

    def __init__(self, max_length: int, message: str | None = None) -> None:
        self.max_length = max_length
        self.message = message or f"Maximum length is {max_length}"

    def __call__(self, value: Any) -> bool:
        if len(str(value)) > self.max_length:
            raise ValidationError(self.message)
        return True


class RangeValidator:
    """Validates numeric value within a range."""

    def __init__(self, min_value: float | None = None, max_value: float | None = None, message: str | None = None) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.message = message

    def __call__(self, value: Any) -> bool:
        val = float(value)
        if self.min_value is not None and val < self.min_value:
            raise ValidationError(self.message or f"Value must be >= {self.min_value}")
        if self.max_value is not None and val > self.max_value:
            raise ValidationError(self.message or f"Value must be <= {self.max_value}")
        return True


class RegexValidator:
    """Validates value against a regex pattern."""

    def __init__(self, pattern: str, message: str = "Invalid format") -> None:
        self.pattern = re.compile(pattern)
        self.message = message

    def __call__(self, value: Any) -> bool:
        if not self.pattern.fullmatch(str(value)):
            raise ValidationError(self.message)
        return True


class Field:
    """Base form field with validation and HTML rendering."""

    def __init__(
        self,
        name: str,
        required: bool = True,
        label: str = "",
        default: Any = None,
        help_text: str = "",
        validators: list[Callable[[Any], bool]] | None = None,
    ) -> None:
        self.name = name
        self.required = required
        self.label = label or name.replace("_", " ").title()
        self.default = default
        self.help_text = help_text
        self.validators = validators or []
        self.error: str | None = None
        self.value: Any = default

    def validate(self, value: Any) -> bool:
        """Validate the field value against all validators."""
        self.value = value
        self.error = None
        if value is None or (isinstance(value, str) and not value.strip()):
            if self.required:
                self.error = f"{self.label} is required"
                return False
            self.value = self.default
            return True
        for validator in self.validators:
            try:
                validator(value)
            except ValidationError as e:
                self.error = e.message
                return False
        return True

    def render(self) -> str:
        """Render the field as an HTML input."""
        val = _html.escape(str(self.value)) if self.value else ""
        error_html = f'<span class="error">{_html.escape(self.error)}</span>' if self.error else ""
        help_html = f'<span class="help">{_html.escape(self.help_text)}</span>' if self.help_text else ""
        return (
            f'<div class="field field-{self.name}">'
            f'<label for="id_{self.name}">{_html.escape(self.label)}</label>'
            f'<input type="text" id="id_{self.name}" name="{self.name}" value="{val}" />'
            f'{help_html}{error_html}'
            f'</div>'
        )


class TextField(Field):
    """Text input field with optional length validators."""

    def __init__(self, name: str, max_length: int | None = None, **kwargs: Any) -> None:
        validators = list(kwargs.pop("validators", []) or [])
        if max_length:
            validators.append(MaxLengthValidator(max_length))
        super().__init__(name, validators=validators, **kwargs)


class EmailField(Field):
    """Email input field with built-in email validation."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        validators = list(kwargs.pop("validators", []) or [])
        validators.append(EmailValidator())
        kwargs.setdefault("help_text", "Enter a valid email address")
        super().__init__(name, validators=validators, **kwargs)


class IntegerField(Field):
    """Integer input field with range validation."""

    def __init__(self, name: str, min_value: int | None = None, max_value: int | None = None, **kwargs: Any) -> None:
        validators = list(kwargs.pop("validators", []) or [])
        if min_value is not None or max_value is not None:
            validators.append(RangeValidator(min_value=min_value, max_value=max_value))
        super().__init__(name, validators=validators, **kwargs)

    def validate(self, value: Any) -> bool:
        if value is not None and value != "":
            try:
                value = int(value)
            except (ValueError, TypeError):
                self.error = f"{self.label} must be a valid integer"
                return False
        self.value = value
        return super().validate(value)

    def render(self) -> str:
        val = str(self.value) if self.value is not None else ""
        error_html = f'<span class="error">{_html.escape(self.error)}</span>' if self.error else ""
        return (
            f'<div class="field field-{self.name}">'
            f'<label for="id_{self.name}">{_html.escape(self.label)}</label>'
            f'<input type="number" id="id_{self.name}" name="{self.name}" value="{_html.escape(val)}" />'
            f'{error_html}</div>'
        )


class FloatField(Field):
    """Floating-point number input field."""

    def __init__(self, name: str, min_value: float | None = None, max_value: float | None = None, **kwargs: Any) -> None:
        validators = list(kwargs.pop("validators", []) or [])
        if min_value is not None or max_value is not None:
            validators.append(RangeValidator(min_value=min_value, max_value=max_value))
        super().__init__(name, validators=validators, **kwargs)

    def validate(self, value: Any) -> bool:
        if value is not None and value != "":
            try:
                value = float(value)
            except (ValueError, TypeError):
                self.error = f"{self.label} must be a valid number"
                return False
        self.value = value
        return super().validate(value)

    def render(self) -> str:
        val = str(self.value) if self.value is not None else ""
        error_html = f'<span class="error">{_html.escape(self.error)}</span>' if self.error else ""
        return (
            f'<div class="field field-{self.name}">'
            f'<label for="id_{self.name}">{_html.escape(self.label)}</label>'
            f'<input type="number" step="any" id="id_{self.name}" name="{self.name}" value="{_html.escape(val)}" />'
            f'{error_html}</div>'
        )


class BooleanField(Field):
    """Boolean checkbox field."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        kwargs.setdefault("required", False)
        super().__init__(name, **kwargs)

    def validate(self, value: Any) -> bool:
        self.error = None
        if isinstance(value, str):
            self.value = value.lower() in ("on", "true", "1", "yes")
        elif isinstance(value, bool):
            self.value = value
        else:
            self.value = bool(value)
        return True

    def render(self) -> str:
        checked = "checked" if self.value else ""
        error_html = f'<span class="error">{_html.escape(self.error)}</span>' if self.error else ""
        return (
            f'<div class="field field-{self.name}">'
            f'<label for="id_{self.name}">{_html.escape(self.label)}</label>'
            f'<input type="checkbox" id="id_{self.name}" name="{self.name}" {checked} />'
            f'{error_html}</div>'
        )


class DateField(Field):
    """Date input field."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

    def validate(self, value: Any) -> bool:
        if value and isinstance(value, str):
            import re
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                self.error = f"{self.label} must be in YYYY-MM-DD format"
                return False
        return super().validate(value)

    def render(self) -> str:
        val = _html.escape(str(self.value)) if self.value else ""
        error_html = f'<span class="error">{_html.escape(self.error)}</span>' if self.error else ""
        return (
            f'<div class="field field-{self.name}">'
            f'<label for="id_{self.name}">{_html.escape(self.label)}</label>'
            f'<input type="date" id="id_{self.name}" name="{self.name}" value="{val}" />'
            f'{error_html}</div>'
        )


class ChoiceField(Field):
    """Select field with predefined choices."""

    def __init__(self, name: str, choices: list[tuple[str, str]] | None = None, **kwargs: Any) -> None:
        self.choices = choices or []
        super().__init__(name, **kwargs)

    def validate(self, value: Any) -> bool:
        self.error = None
        if value and self.choices:
            valid_values = [c[0] for c in self.choices]
            if value not in valid_values:
                self.error = f"{self.label} must be one of: {', '.join(valid_values)}"
                return False
        return super().validate(value)

    def render(self) -> str:
        options_html = '<option value="">-- Select --</option>'
        for val, label in self.choices:
            selected = "selected" if str(self.value) == val else ""
            options_html += f'<option value="{_html.escape(val)}" {selected}>{_html.escape(label)}</option>'
        error_html = f'<span class="error">{_html.escape(self.error)}</span>' if self.error else ""
        return (
            f'<div class="field field-{self.name}">'
            f'<label for="id_{self.name}">{_html.escape(self.label)}</label>'
            f'<select id="id_{self.name}" name="{self.name}">{options_html}</select>'
            f'{error_html}</div>'
        )


class MultipleChoiceField(Field):
    """Multi-select field with checkboxes."""

    def __init__(self, name: str, choices: list[tuple[str, str]] | None = None, **kwargs: Any) -> None:
        self.choices = choices or []
        kwargs.setdefault("required", False)
        super().__init__(name, **kwargs)

    def validate(self, value: Any) -> bool:
        self.error = None
        if isinstance(value, list) and self.choices:
            valid_values = {c[0] for c in self.choices}
            for v in value:
                if v not in valid_values:
                    self.error = f"Invalid choice: {v}"
                    return False
        return True

    def render(self) -> str:
        checkboxes = ""
        selected = self.value if isinstance(self.value, list) else []
        for val, label in self.choices:
            checked = "checked" if val in selected else ""
            checkboxes += (
                f'<label><input type="checkbox" name="{self.name}" value="{_html.escape(val)}" {checked} />'
                f' {_html.escape(label)}</label> '
            )
        error_html = f'<span class="error">{_html.escape(self.error)}</span>' if self.error else ""
        return (
            f'<div class="field field-{self.name}">'
            f'<label>{_html.escape(self.label)}</label>'
            f'{checkboxes}{error_html}</div>'
        )


class FileField(Field):
    """File upload field with extension and size validation."""

    def __init__(self, name: str, allowed_extensions: list[str] | None = None, max_size: int = 10 * 1024 * 1024, **kwargs: Any) -> None:
        self.allowed_extensions = allowed_extensions or []
        self.max_size = max_size
        super().__init__(name, **kwargs)

    def validate(self, value: Any) -> bool:
        self.error = None
        if value and isinstance(value, dict):
            filename = value.get("filename", "")
            content = value.get("content", b"")
            if self.allowed_extensions:
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext not in self.allowed_extensions:
                    self.error = f"File extension must be one of: {', '.join(self.allowed_extensions)}"
                    return False
            if len(content) > self.max_size:
                self.error = f"File size must not exceed {self.max_size} bytes"
                return False
        return super().validate(value)


class PasswordField(Field):
    """Password input field with minimum length validation."""

    def __init__(self, name: str, min_length: int = 8, **kwargs: Any) -> None:
        validators = list(kwargs.pop("validators", []) or [])
        validators.append(MinLengthValidator(min_length, f"Password must be at least {min_length} characters"))
        kwargs.setdefault("help_text", f"Minimum {min_length} characters")
        super().__init__(name, validators=validators, **kwargs)

    def render(self) -> str:
        error_html = f'<span class="error">{_html.escape(self.error)}</span>' if self.error else ""
        help_html = f'<span class="help">{_html.escape(self.help_text)}</span>' if self.help_text else ""
        return (
            f'<div class="field field-{self.name}">'
            f'<label for="id_{self.name}">{_html.escape(self.label)}</label>'
            f'<input type="password" id="id_{self.name}" name="{self.name}" />'
            f'{help_html}{error_html}</div>'
        )


class Form:
    """Form class binding data to fields with validation and HTML rendering."""

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._data = data or {}
        self._fields: dict[str, Field] = {}
        self._cleaned: dict[str, Any] | None = None
        self._is_bound = bool(self._data)
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name)
            if isinstance(attr, Field):
                field_instance = type(attr)(
                    name=attr.name or attr_name,
                    required=attr.required,
                    label=attr.label,
                    default=attr.default,
                    help_text=attr.help_text,
                    validators=list(attr.validators),
                )
                setattr(self, attr_name, field_instance)
                self._fields[field_instance.name] = field_instance

    def is_valid(self) -> bool:
        """Validate all fields and return overall validity."""
        self._cleaned = {}
        all_valid = True
        for name, field_obj in self._fields.items():
            raw = self._data.get(name, field_obj.default)
            if not field_obj.validate(raw):
                all_valid = False
            else:
                self._cleaned[name] = field_obj.value
        if all_valid:
            self._cleaned_data()
        return all_valid

    def _cleaned_data(self) -> None:
        if self._cleaned is None:
            self._cleaned = {}

    @property
    def errors(self) -> dict[str, list[str]]:
        """Return dict of field_name -> list of error messages."""
        result: dict[str, list[str]] = {}
        for name, field_obj in self._fields.items():
            if field_obj.error:
                result[name] = [field_obj.error]
        return result

    @property
    def cleaned_data(self) -> dict[str, Any]:
        """Return validated and cleaned data."""
        return self._cleaned or {}

    def as_html(self) -> str:
        """Render the entire form as HTML."""
        parts = ['<form method="post" enctype="multipart/form-data">']
        for _name, field_obj in self._fields.items():
            parts.append(field_obj.render())
        parts.append('<button type="submit">Submit</button>')
        parts.append("</form>")
        return "\n".join(parts)


class ModelForm(Form):
    """Auto-generate form fields from a dataclass model."""

    _field_type_map: dict[type, type[Field]] = {}

    @classmethod
    def register_type(cls, python_type: type, field_type: type[Field]) -> None:
        cls._field_type_map[python_type] = field_type

    @classmethod
    def from_model(cls, model_class: type, data: dict[str, Any] | None = None) -> ModelForm:
        """Create a ModelForm dynamically from a dataclass."""
        import dataclasses
        fields_dict: dict[str, Field] = {}
        for f in dataclasses.fields(model_class):
            field_type = cls._field_type_map.get(f.type, TextField)
            kwargs: dict[str, Any] = {"name": f.name, "label": f.name.replace("_", " ").title()}
            if f.default is not dataclasses.MISSING:
                kwargs["default"] = f.default
            if f.default is not dataclasses.MISSING:
                kwargs["required"] = False
            fields_dict[f.name] = field_type(**kwargs)

        class DynamicModelForm(ModelForm):
            pass

        for fname, field_obj in fields_dict.items():
            setattr(DynamicModelForm, fname, field_obj)

        return DynamicModelForm(data=data or {})


ModelForm.register_type(str, TextField)
ModelForm.register_type(int, IntegerField)
ModelForm.register_type(float, FloatField)
ModelForm.register_type(bool, BooleanField)
