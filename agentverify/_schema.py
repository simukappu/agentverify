"""The small part of JSON Schema that ``assert_tool_calls_allowed`` reads for tool arguments.

Keywords: ``type``, ``enum``, ``const``, ``minimum``, ``maximum``, ``minLength``, ``maxLength``, ``pattern``, ``required``, ``properties``, ``additionalProperties``, and the annotations ``title``, ``description``, ``default``, ``examples``, ``$schema``. Any other keyword raises ``ValueError``, so a condition that would not be checked is not silently ignored.

Stricter than JSON Schema in one place: ``integer`` takes no value written as a float, not even ``3.0``. Booleans are never numbers, ``number`` takes no NaN or infinity, and ``enum`` / ``const`` compare types too (``True`` is not ``1``, ``1.0`` is not ``1``).
"""

from __future__ import annotations

import math
import re
from typing import Any

_KEYWORDS = {"type", "enum", "const", "minimum", "maximum", "minLength", "maxLength", "pattern", "required", "properties", "additionalProperties"}
_ANNOTATIONS = {"title", "description", "default", "examples", "$schema"}
_TYPES = {"string", "integer", "number", "boolean", "object", "array", "null"}


def check_schema(schema: Any, where: str) -> None:
    """Raise ``ValueError`` for a schema this module cannot check in full."""
    if not isinstance(schema, dict):
        raise ValueError(f"{where}: a schema must be a dict, got {schema!r}")
    unknown = sorted(set(schema) - _KEYWORDS - _ANNOTATIONS)
    if unknown:
        raise ValueError(f"{where}: unsupported schema keyword(s) {unknown}; supported: {sorted(_KEYWORDS)}")
    types = schema.get("type")
    names = [] if types is None else [types] if isinstance(types, str) else types if isinstance(types, list) else [types]
    for name in names:
        if name not in _TYPES:
            raise ValueError(f"{where}: unknown type {name!r}")

    def bad(keyword: str, expected: str) -> ValueError:
        return ValueError(f"{where}: {keyword} must be {expected}, got {schema[keyword]!r}")

    for keyword in ("minimum", "maximum"):
        if keyword in schema and (isinstance(schema[keyword], bool) or not isinstance(schema[keyword], (int, float))):
            raise bad(keyword, "a number")
    for keyword in ("minLength", "maxLength"):
        if keyword in schema and (isinstance(schema[keyword], bool) or not isinstance(schema[keyword], int) or schema[keyword] < 0):
            raise bad(keyword, "a non-negative integer")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise bad("enum", "a list")
    if "required" in schema and not (isinstance(schema["required"], list) and all(isinstance(n, str) for n in schema["required"])):
        raise bad("required", "a list of names")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (re.error, TypeError) as error:
            raise bad("pattern", f"a regular expression ({error})") from None
    if "properties" in schema and not isinstance(schema["properties"], dict):
        raise bad("properties", "a dict of argument names to schemas")
    for name, sub in (schema.get("properties") or {}).items():
        check_schema(sub, f"{where}.{name}")
    extra = schema.get("additionalProperties")
    if extra is not None and not isinstance(extra, bool):
        check_schema(extra, f"{where}.additionalProperties")


def _is_type(value: Any, name: str) -> bool:
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if isinstance(value, bool):
        return False
    if name == "integer":
        return isinstance(value, int)
    return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))  # number


def _same(a: Any, b: Any) -> bool:
    """Equal and of the same type, all the way down."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return bool(a == b)


def violations(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Why *value* does not meet *schema*, one reason per problem (empty when it does)."""
    here = path or "arguments"
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if not any(_is_type(value, name) for name in names):
            return [f"{here}: expected {' or '.join(names)}, got {value!r}"]
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{here}: {value!r} is not a finite number"]
    reasons: list[str] = []
    if "const" in schema and not _same(value, schema["const"]):
        reasons.append(f"{here}: must be {schema['const']!r}, got {value!r}")
    if "enum" in schema and not any(_same(value, option) for option in schema["enum"]):
        reasons.append(f"{here}: {value!r} is not one of {schema['enum']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            reasons.append(f"{here}: {value!r} is below the minimum {schema['minimum']!r}")
        if "maximum" in schema and value > schema["maximum"]:
            reasons.append(f"{here}: {value!r} is above the maximum {schema['maximum']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            reasons.append(f"{here}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            reasons.append(f"{here}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            reasons.append(f"{here}: does not match pattern {schema['pattern']!r}")
    if isinstance(value, dict):
        for name in schema.get("required") or []:
            if name not in value:
                reasons.append(f"{here}: missing required {name!r}")
        properties = schema.get("properties") or {}
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            where = f"{path}.{name}" if path else name
            if name in properties:
                reasons += violations(item, properties[name], where)
            elif extra is False:
                reasons.append(f"{where}: not an allowed argument")
            elif isinstance(extra, dict):
                reasons += violations(item, extra, where)
    return reasons
