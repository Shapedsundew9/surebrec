"""Surebrec is a utility to randomly generate data conforming to a Cerberus Validation schema.

Surebrec is Cerberus backwards.

Limitiations
------------
    1. Generating from regular expressions is tricky: surebrec will try but may fail.
    2. If regular expressions are used in the validator they must be defined so as not to break other rules e.g. length.
    3. dict type returns a dict not a Mapping
    4. list type returns a list not a Sequence
    5. Only python 3 types are supported
    6. 'check_with' constraints are not supported.
    7.

Extensions
----------
    1. Supports 'uuid' type as python UUID object

"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
from logging import DEBUG, Logger, NullHandler, getLogger
from pprint import pformat
from random import choice, choices, getrandbits, randint, seed, uniform
from string import printable
from types import GeneratorType
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    LiteralString,
    Type,
    TypedDict,
    TypeVar,
    Hashable
)
from uuid import UUID

from cerberus import Validator
from exrex import getone
from numpy.random import normal
from numpy.random import seed as np_seed


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)

# Types
# Value range types
T = TypeVar("T", int, float, date, datetime)


class ValueLimits(TypedDict, Generic[T]):
    """Specify a valid range for the mapped Cerberus type."""

    min: T
    max: T


class SequenceLimits(TypedDict):
    """Restrictions on sequence generation if not specified in the schema."""

    maxlength: int


class ContainerLimits(TypedDict):
    """Restrictions on container generation if not specified in the schema."""

    maxelements: int


class GeneralLimits(TypedDict):
    """General limits for generation of data."""

    rare_events_one_in_x: int
    required_one_in_x: int
    max_nested_containers: int
    random_schema_maxlength: int
    random_length_stddev: float 

class Limits(TypedDict):
    """Limits for generation of data."""

    general: GeneralLimits
    boolean: dict[LiteralString, Any]
    binary: SequenceLimits
    date: ValueLimits[date]
    datetime: ValueLimits[datetime]
    number: ValueLimits[float]
    float: ValueLimits[float]
    integer: ValueLimits[int]
    list: ContainerLimits
    set: ContainerLimits
    string: SequenceLimits
    uuid: dict[LiteralString, Any]
    dict: ContainerLimits


# LIMITS defines the limits of unbounded configurations.
# NOTE: That the 'number' type inherits from 'float' or 'int' as appropriate.
# In the 'general' section:
#   'rare_events_one_in_x': Chance of selecting a 'empty', 'null'.
#   'required_one_in_x': Chance of selecting a 'not required' field.
#   'max_nested_containers': When randomly generating a schema the max depth of nested containers.
#   'random_schema_maxlength': The max number of fields and length or sequences in a random schema unless otherwise specified.
MAX_LENGTH = 32
MAX_ELEMENTS = 128
LIMITS: Limits = {
    "general": {
        "rare_events_one_in_x": 1000,
        "required_one_in_x": 2,
        "max_nested_containers": 3,
        "random_schema_maxlength": 16,
        "random_length_stddev": 0.25
    },
    "boolean": {},
    "binary": {"maxlength": MAX_LENGTH},
    "date": {"min": date.min, "max": date.max},
    "datetime": {"min": datetime.min, "max": datetime.max},
    "dict": {"maxelements": MAX_ELEMENTS},
    "float": {"min": -1.7976931348623157e308, "max": 1.7976931348623157e308},
    "integer": {"min": -(2**63), "max": 2**63 - 1},
    "number": {"min": -1.7976931348623157e308, "max": 1.7976931348623157e308},
    "list": {"maxelements": MAX_ELEMENTS},
    "set": {"maxelements": MAX_ELEMENTS},
    "string": {"maxlength": MAX_LENGTH},
    "uuid": {},
}

DEFAULT_KEYSRULES: dict[str, list[str]] = {
    "type": [
        "boolean",
        "integer",
        "float",
        "date",
        "datetime",
        "number",
        "string",
        "uuid",
    ],
}


def _get_length(minlength: int, maxlength: int) -> int:
    """Return a random length between minlength and maxlength with a gaussian distribution."""
    if maxlength == minlength:
        return minlength
    delta: int = maxlength - minlength
    return (abs(int(normal(0, delta * LIMITS["general"]["random_length_stddev"]))) % delta) + minlength


def _rare_event() -> bool:
    """True if a rare event occurs."""
    return not randint(0, LIMITS["general"]["rare_events_one_in_x"] - 1)


def _required() -> bool:
    """True if a rare event occurs."""
    return not randint(0, LIMITS["general"]["required_one_in_x"] - 1)


def _generate_base(constraints: dict) -> tuple[bool, Any]:
    """Common generation operations for every type.

    Args
    ----
    constraints: Cerberus value definition schema.
    typ: The class of the type being generated.

    Returns
    -------
    (value_set: bool, value: Any) If value_set is True value represents the value generated.
    """
    coerce: Callable = constraints.get("coerce", lambda x: x)
    if "allowed" in constraints:
        if not constraints["allowed"]:
            raise ValueError("Allowed values list is empty.")
        _logger.debug("Value chosen from allowed values.")
        return (True, coerce(choice(constraints["allowed"])))
    if "default" in constraints and _rare_event():
        _logger.debug("Value set to the default.")
        return (
            (True, coerce(constraints["default"]))
            if constraints["default"] is not None
            else (True, None)
        )
    if constraints.get("nullable", False) and _rare_event():
        _logger.debug("Value set to None.")
        return (True, None)
    return (False, None)


def _generate_boolean(constraints: dict, _: int = 0) -> bool | None:
    _logger.debug("Generating bool value.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    return common_tuple[1] if common_tuple[0] else bool(randint(0, 1))


def _generate_binary(constraints: dict, _: int = 0) -> bytes | bytearray | None:
    _logger.debug("Generating binary value.")
    retype: Type[bytes] | Type[bytearray] = choice((bytes, bytearray))
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    minlength: int = constraints.get("minlength", 0)
    maxlength: int = constraints.get("maxlength", LIMITS["binary"]["maxlength"])
    return retype(
        bytearray(getrandbits(8) for _ in range(_get_length(minlength, maxlength)))
    )


def _generate_date(constraints: dict, _: int = 0) -> date:
    _logger.debug("Generating date value.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    minval: date = constraints.get("min", LIMITS["date"]["min"])
    maxval: date = constraints.get("max", LIMITS["date"]["max"])
    return minval + timedelta(randint(0, (maxval - minval).days))


def _generate_datetime(constraints: dict, _: int = 0) -> datetime | None:
    _logger.debug("Generating datetime value.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    minval: datetime = constraints.get("min", LIMITS["datetime"]["min"])
    maxval: datetime = constraints.get("max", LIMITS["datetime"]["max"])
    delta: timedelta = maxval - minval
    days: int = randint(0, delta.days)
    seconds: int = randint(0, delta.seconds)
    microseconds: int = randint(0, delta.microseconds)
    return minval + timedelta(days, seconds, microseconds)


def _generate_dict(constraints: dict, depth: int = 0) -> dict | None:
    # As per https://docs.python-cerberus.org/en/stable/validation-rules.html#empty
    # emptiness is not checked if not defined thus only allowed by the validation (generation) rules
    _logger.debug("Generating dictionary container.")
    if constraints.get("empty", False) and _rare_event():
        _logger.debug("dict set to empty.")
        return {}
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    minlength: int = constraints.get("minlength", 0)
    maxlength: int = constraints.get(
        "maxlength", LIMITS["general"]["random_schema_maxlength"]
    )
    if "schema" in constraints:
        _logger.debug("dict has a schema.")
        schema: dict[str, Any] = constraints["schema"]
        if "valuesrules" in constraints:
            _logger.debug("Merging valuerules with schema definitions.")
            for definition in schema.values():
                definition.update(constraints["valuesrules"])
    else:
        _logger.debug("dict has no schema.")
        schema = _random_dict_schema(
            constraints.get("keysrules"),
            constraints.get("valuesrules"),
            minlength,
            maxlength,
            depth + 1,
        )
    _dict: dict = {}
    for field, definition in schema.items():
        required: bool = (
            definition.get("required", False)
            or schema.get("require_all", False)
            or (not definition.get("required", False) and _required())
            or definition.get("meta", {}).get("defined", False)
        )
        readonly: bool = definition.get("readonly", False) and not (
            "default" in definition or "default_setter" in definition
        )
        if _LOG_DEBUG:
            _logger.debug(
                f"dict field '{field}' required = {required}, readonly = {readonly}."
            )
        if required and not readonly:
            _type: str = (
                definition["type"]
                if not isinstance(definition["type"], list)
                else choice(definition["type"])
            )
            _dict[field] = TYPE_GENERATION[_type](definition, depth + 1)
    return _dict


def _generate_float(constraints: dict, _: int = 0) -> float | None:
    _logger.debug("Generating float value.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    minval: float = constraints.get("min", LIMITS["float"]["min"])
    maxval: float = constraints.get("max", LIMITS["float"]["max"])
    return uniform(minval, maxval)


def _generate_integer(constraints: dict, _: int = 0) -> int | None:
    _logger.debug("Generating integer value.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    minval: int = constraints.get("min", LIMITS["integer"]["min"])
    maxval: int = constraints.get("max", LIMITS["integer"]["max"])
    return randint(minval, maxval)


def _generate_list(constraints: dict, depth: int = 0) -> list | None:
    # As per https://docs.python-cerberus.org/en/stable/validation-rules.html#empty
    # emptiness is not checked if not defined thus only allowed by the validation (generation) rules
    if constraints.get("empty", False) and _rare_event():
        _logger.debug("list set to empty.")
        return []
    _logger.debug("Generating list container.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    if "items" in constraints:
        _list: list[Any] = []
        _logger.debug("list generated from item constraints")
        for schema in constraints["items"]:
            _type: str = (
                schema.get("type", "string")
                if not isinstance(schema.get("type", "string"), list)
                else choice(schema["type"])
            )
            _list.append(TYPE_GENERATION[_type](schema, depth + 1))
        return _list
    if "schema" in constraints:
        _logger.debug("list randomly determined from schema.")
        schema: dict[str, Any] = constraints["schema"]
    else:
        _logger.debug("list randomly determined from random schema.")
        schema = _random_list_schema(depth + 1)
    _list = []
    minlength: int = constraints.get("minlength", 0)
    maxlength: int = constraints.get(
        "maxlength", LIMITS["general"]["random_schema_maxlength"]
    )
    num: int = _get_length(minlength, maxlength)
    if _LOG_DEBUG:
        _logger.debug(f"Randomly choosing {num} elements.")
    for _ in range(num):
        _type = (
            schema["type"]
            if not isinstance(schema["type"], list)
            else choice(schema["type"])
        )
        _list.append(TYPE_GENERATION[_type](schema, depth + 1))
    return _list


def _generate_number(constraints: dict, depth: int = 0) -> float | int | None:
    return choice((_generate_integer, _generate_float))(constraints, depth)


def _generate_set(constraints: dict, depth: int = 0) -> set | None:
    # As per https://docs.python-cerberus.org/en/stable/validation-rules.html#empty
    # emptiness is not checked if not defined thus only allowed by the validation (generation) rules
    if constraints.get("empty", False) and _rare_event():
        return set()
    _logger.debug("Generating set container.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    minlength: int = constraints.get("minlength", 0)
    maxlength: int = constraints.get(
        "maxlength", LIMITS["general"]["random_schema_maxlength"]
    )
    schema: dict[str, Any] = (
        constraints["schema"]
        if "schema" in constraints
        else _random_list_schema(depth + 1, True)
    )
    _set: set[Any] = set()
    for _ in range(_get_length(minlength, maxlength)):
        _type: str = (
            schema["type"]
            if not isinstance(schema["type"], list)
            else choice(schema["type"])
        )
        _set.add(TYPE_GENERATION[_type](schema, depth + 1))
    return _set


def _generate_string(constraints: dict, _: int = 0) -> str | None:
    # As per https://docs.python-cerberus.org/en/stable/validation-rules.html#empty
    # emptiness is not checked if not defined thus only allowed by the validation (generation) rules
    if constraints.get("empty", False) and _rare_event():
        return ""
    _logger.debug("Generating string value.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    if "regex" in constraints:
        return getone(constraints["regex"])
    minlength: int = constraints.get("minlength", 0)
    maxlength: int = constraints.get("maxlength", LIMITS["string"]["maxlength"])
    return "".join(choices(printable, k=_get_length(minlength, maxlength)))


def _generate_uuid(constraints: dict, _: int = 0) -> UUID | None:
    _logger.debug("Generating UUID value.")
    common_tuple: tuple[bool, Any] = _generate_base(constraints)
    if common_tuple[0]:
        return common_tuple[1]
    return UUID(bytes=bytes((getrandbits(8) for _ in range(16))))


TYPE_GENERATION: dict[str, Callable[(...), Any | None]] = {
    "boolean": _generate_boolean,
    "binary": _generate_binary,
    "date": _generate_date,
    "datetime": _generate_datetime,
    "dict": _generate_dict,
    "float": _generate_float,
    "integer": _generate_integer,
    "list": _generate_list,
    "number": _generate_number,
    "set": _generate_set,
    "string": _generate_string,
    "uuid": _generate_uuid,
}
TYPES: list[str] = list(TYPE_GENERATION.keys())[0:11]


def _type_choices(types: list[str], depth=0, hashable=False) -> dict[str, list[str]]:
    if depth == LIMITS["general"]["max_nested_containers"]:
        if "dict" in types:
            types.remove("dict")
        if "list" in types:
            types.remove("list")
        if "set" in types:
            types.remove("set")
    if hashable:
        if "binary" in types:
            types.remove("binary")
        if "dict" in types:
            types.remove("dict")
        if "list" in types:
            types.remove("list")
        if "set" in types:
            types.remove("set")
    return {"type": choices(types, k=randint(1, len(types)))}


def _random_dict_schema(
    keysrules: dict[str, Any] | None,
    valuesrules: dict[str, Any] | None,
    minlength: int,
    maxlength: int,
    depth=0,
) -> dict[str, dict[str, Any]]:
    if keysrules is None:
        _logger.debug("Randomly generating dict keyrules.")
        keysrules = DEFAULT_KEYSRULES
        keysrules["minlength"] = 4
        keysrules["maxlength"] = LIMITS["general"]["random_schema_maxlength"]
    keys: list[Any] = []
    for _ in range(_get_length(minlength, maxlength)):
        _type: str = (
            keysrules["type"]
            if not isinstance(keysrules["type"], list)
            else choice(keysrules["type"])
        )
        keys.append(TYPE_GENERATION[_type](keysrules))
    if valuesrules is None:
        _logger.debug("Randomly generating dict valuerules.")
        # Only Cerberus native types
        return dict(zip(keys, [_type_choices(list(TYPES), depth) for _ in keys]))
    else:
        return dict(zip(keys, [valuesrules] * len(keys)))


def _random_list_schema(depth=0, hashable=False) -> dict[str, list[str]]:
    """Randomly determine the types that may appear in a list."""
    return _type_choices(list(TYPE_GENERATION.keys()), depth, hashable)


def _strip_check_with(container: dict | list) -> None:
    """Surebrec does not support 'check_with' behaviour."""
    if isinstance(container, dict):
        if "check_with" in container:
            del container["check_with"]
        container = list(container.values())
    for value in filter(lambda x: isinstance(x, (dict, list)), container):
        _strip_check_with(value)


def _strip_meta(schema: dict) -> None:
    """Surebrec uses the meta field to point to dependencies."""
    for _, definition in schema.items():
        definition["meta"] = {}
        typ = definition.get("type")
        if (
            (isinstance(typ, (list, tuple, set)) and "dict" in typ) or typ == "dict"
        ) and ("schema" in definition):
            _strip_meta(definition["schema"])


def is_iterable(obj: Any) -> bool:
    """True if obj is iterable."""
    if isinstance(obj, GeneratorType):
        return True
    try:
        _: Any = iter(obj)
    except TypeError:
        return False
    return True


def _nest_coersions(to_wrap: Callable, wrappers: list[Callable]) -> Callable:
    if wrappers:
        return _nest_coersions(lambda x: wrappers.pop(0)(to_wrap(x)), wrappers)
    return to_wrap


def _callable_coersion(coersion: str | Callable, validator: Validator) -> Callable:
    if isinstance(coersion, str):
        return getattr(validator, "_normalize_coerce_" + coersion)
    return coersion


def _define_coercers(container: dict | list, validator: Validator) -> None:
    """Replace any method string coercers with function calls and iterable with a nested coersion call.

    The 'coerce' value of a schema may be callable or string or an iterable of both.
    The rules for such are here: https://docs.python-cerberus.org/en/stable/normalization-rules.html#value-coercion
    Note that 'default' & 'allowed' values are also coersed.
    """
    if isinstance(container, dict):
        if "coerce" in container:
            if not isinstance(container["coerce"], str) and is_iterable(
                container["coerce"]
            ):
                coersions: list[Callable] = [
                    _callable_coersion(coersion, validator)
                    for coersion in container["coerce"]
                ]
                if len(coersions) > 1:
                    coersion: Callable = _nest_coersions(coersions[0], coersions[1:])
                else:
                    coersion: Callable = coersions[0]
            else:
                coersion = _callable_coersion(container["coerce"], validator)
            container["coerce"] = coersion
        container = list(container.values())
    for value in filter(lambda x: isinstance(x, (dict, list)), container):
        _define_coercers(value, validator)


def _is_list(definition) -> bool:
    """True if the definition is a list."""
    assert (
        "type" in definition
    ), "Schema defined but no type (list or dict) specified. Impossible to resolve."
    if isinstance(definition["type"], list):
        if "list" in definition["type"]:
            assert (
                "dict" not in definition["type"]
            ), "Ambiguous schema type specified: Both list and dict defined."
            return True
        return False
    # This allows a non-list type, even if it is not a dict to have a schema.
    return definition["type"] == "list"


def _is_dict(definition: dict) -> bool:
    """True if the definition is a dict."""
    assert (
        "type" in definition
    ), "Schema defined but no type (list or dict) specified. Impossible to resolve."
    if isinstance(definition["type"], dict):
        if "dict" in definition["type"]:
            assert (
                "list" not in definition["type"]
            ), "Ambiguous schema type specified: Both list and dict defined."
            return True
        return False
    # This allows a non-list type, even if it is not a dict to have a schema.
    return definition["type"] == "dict"


def _define_structure(definition: dict, validator: Validator) -> dict:
    """Recursively expand the structure of a field.

    Args
    ----
    definition: The defintion of a field or schema.
    validator: The validator

    Returns
    -------
    A schema with all the registry references expanded to thier definitions.
    """
    if isinstance(definition, str):
        if _LOG_DEBUG:
            _logger.debug(f"Rules definition pulled from registry '{definition}'.")
        rdef = validator.rules_set_registry.get(definition)  # type: ignore
        assert rdef is not None, "No such rules exist in the registry."
        if _LOG_DEBUG:
            _logger.debug(f"Rules are:\n{pformat(rdef)}.")
    else:
        rdef = definition
    if _LOG_DEBUG:
        _logger.debug(f"Assessing definition structure {list(rdef.keys())}")

    # TODO: Need to consider recursive oneof_* definitions. Presently we choose a random oneof_* definition
    # and then expand it. This is not correct. We choose from the oneof_* for each item in the schema.
    if "schema" in rdef:
        if _is_list(rdef):
            if isinstance(rdef["schema"], str):
                if _LOG_DEBUG:
                    _logger.debug(
                        f"Rules definition pulled from registry '{rdef['schema']}'."
                    )
                rdef["schema"] = validator.rules_set_registry.get(rdef["schema"])  # type: ignore
                assert rdef is not None, "No such rules exist in the registry."
                if _LOG_DEBUG:
                    _logger.debug(f"Rules are:\n{pformat(rdef)}.")
            rdef["schema"] = _define_structure(rdef["schema"], validator)
        elif _is_dict(rdef):
            if isinstance(rdef["schema"], str):
                if _LOG_DEBUG:
                    _logger.debug(
                        f"Schema definition pulled from registry '{rdef['schema']}'."
                    )
                rdef["schema"] = validator.schema_registry.get(rdef["schema"])  # type: ignore
                if _LOG_DEBUG:
                    _logger.debug(
                        f"Schema is\n{pformat(rdef['schema'], indent=4, sort_dicts=True)}"
                    )
            rdef["schema"] = _define_structure(rdef["schema"], validator)
            for key in tuple(rdef["schema"].keys()):
                rdef["schema"][key] = _define_structure(rdef["schema"][key], validator)
                if rdef["schema"][key] is None:
                    del rdef["schema"][key]

    # Dynamic definition of either keyrules or valuerules
    if "valuesrules" in rdef and "keysrules" in rdef:
        _logger.debug("keysrules & valuesrules defined for dict.")
        if (any(k.startswith("oneof") or k.startswith("anyof") for k in rdef["keysrules"]) or
                any(k.startswith("oneof") or k.startswith("anyof") for k in rdef["valuesrules"])):
            _logger.debug("Dynamic definition of keysrules & valuesrules.")
            # If there are keysrules or valuesrules then there may be multiple items in the schema.
            # each of which must fit these rules not an instance of them.
            # First we determine the length of the schema.
            minlength: int = definition.get("minlength", 0)
            maxlength: int = definition.get("maxlength", LIMITS["general"]["random_schema_maxlength"])
            if "schema" not in rdef:
                rdef["schema"] = {}
            for _ in range(_get_length(minlength, maxlength)):
                keysrules = _define_structure(rdef["keysrules"], validator)
                keysrules.update({"required": True})
                t_validator = Validator({"key": keysrules})  # type: ignore
                gkey = generate(t_validator, 1, randint(0, 2**31 - 1), randint(0, 2**31 - 1), False)[0].get("key")
                if not isinstance(gkey, Hashable) or gkey is None:
                    raise ValueError(f"Generated key '{gkey}' is not hashable. Check 'keyrules' definition.")
                rdef["schema"][gkey] = _define_structure(deepcopy(rdef["valuesrules"]), validator)
            del rdef["keysrules"]
            del rdef["valuesrules"]

    # Definitions of keys rules & valuesrules are static
    if "keysrules" in rdef:
        rdef["keysrules"] = _define_structure(rdef["keysrules"], validator)
    if "valuesrules" in rdef:
        rdef["valuesrules"] = _define_structure(rdef["valuesrules"], validator)

    # Select a value from anyof_*, oneof_* rules
    for k in rdef:
        if k.startswith("anyof_") or k.startswith("oneof_"):
            rdef[k[6:]] = choice(rdef[k])
            del rdef[k]

    # Pure anyof & oneof values are merged into rdef
    if "anyof" in rdef or "oneof" in rdef:
        key: Literal["anyof", "oneof"] = ("anyof", "oneof")["oneof" in rdef]
        if _LOG_DEBUG:
            _logger.debug(
                f"'{key}' defined:\n{pformat(rdef, indent=4, sort_dicts=True)}"
            )
        one_def = deepcopy(rdef[key])
        del rdef[key]
        rdef.update(choice(one_def))
        rdef = _define_structure(rdef, validator)
        if _LOG_DEBUG:
            _logger.debug(
                f"Definition updated to:\n{pformat(rdef, indent=4, sort_dicts=True)}"
            )

    # TODO: all_of & none_of ... but not sure how yet
    if "items" in rdef:
        if _LOG_DEBUG:
            _logger.debug(f"'items' defined:\n{pformat(rdef['items'])}")
        rdef["items"] = [_define_structure(item, validator) for item in rdef["items"]]
        if _LOG_DEBUG:
            _logger.debug(
                f"Definition items expanded to:\n{pformat(rdef['items'], indent=4, sort_dicts=True)} "
            )
    return rdef


def _find_dep(schema: dict, validator: Validator, dep: str) -> dict:
    """Find the dependency in the validator/schema and return its definition.

    dep could be referenced relative to the current schema, or it could be defined
    as relative to the root of the validator schema.

    Args
    ----
    schema: An expanded schema i.e. all registry references have been expanded.
    validator: The associated validator for the schema.
    dep: The name of the dependency field to find.

    Returns
    -------
    The definition dict of dep.
    """
    _logger.debug(f"Looking for dependency '{dep}'")
    if dep in schema:
        _logger.debug("Found dependency in current schema.")
        return schema[dep]
    assert len(dep) > 0, "'' key must be in schema."
    if dep[0] == "^":
        assert (
            len(dep) > 1 and dep[1] != "^"
        ), f"Dependency '{dep}' does not exist in schema."
        if dep == "^":
            _logger.debug("Corner case dependency key '' in root")
            # Corner case of '' key in the root
            return validator.schema[dep]  # type: ignore
        _logger.debug("Searching from root.")
        path = dep[1:].split(".")
        lookup_schema = validator.schema  # type: ignore
    else:
        _logger.debug("Searching in current schema.")
        path = dep.split(".")
        lookup_schema = schema
    for step in path[:-1]:
        assert step in lookup_schema, f"Field '{step}' of '{dep}' not found in schema."
        assert (
            "schema" in lookup_schema[step]
        ), f"Field '{step}' of '{dep}' is not a sub-schema."
        lookup_schema = lookup_schema[step]["schema"]
    _logger.debug("Found dependency.")
    return lookup_schema[path[-1]]


def _generate_deps(schema: dict, validator: Validator) -> None:
    """Recurse the schema and add references to dependent fields as needed.

    If dependencies are found they are added into the 'meta' field with the structure:
    list[tuple[schema/rules set, allowed values]]

    The schema/rules set is a direct reference to the dependency definition in the schema.
    Allowed values is None if the dependencies were not defined with a mapping.

    Args
    ----
    schema: An expanded schema i.e. all registry references have been expanded.
    validator: The associated validator for the schema.
    """
    for field, definition in schema.items():
        _logger.debug(f"Checking '{field}' for dependencies.")
        if "dependencies" in definition:
            _logger.debug(f"'{field}' has a dependencies.")
            deps = definition["dependencies"]
            if isinstance(deps, str):
                definition["meta"].setdefault("deps", []).append(
                    (_find_dep(schema, validator, deps), None)
                )
            elif isinstance(deps, (list, tuple)):
                for dep in deps:
                    definition["meta"].setdefault("deps", []).append(
                        (_find_dep(schema, validator, dep), None)
                    )
            elif isinstance(deps, dict):
                for dep, allowed in deps.items():
                    definition["meta"].setdefault("deps", []).append(
                        (_find_dep(schema, validator, dep), allowed)
                    )
        typ = definition.get("type")
        if (
            (isinstance(typ, (list, tuple, set)) and "dict" in typ) or typ == "dict"
        ) and ("schema" in definition):
            _generate_deps(definition["schema"], validator)


def generate(
    validator: Validator,
    num: int = 1000,
    s_seed: int | None = None,
    v_seed: int | None = None,
    validate: bool = False,
) -> list[dict[Any, Any]]:
    """Generate data conformant to validator schema.

    Args
    ----
    validator: A Cerberus Validator
    num: The number of records to return
    s_seed: Random number generater seed for structure reproducability
    v_seed: Random number generater seed for value reproducability
    validate: User the supplied validator to validate the output.

    Returns
    -------
    list of records that comply with validator.
    """
    _s_seed: int = randint(0, 2**31 - 1) if s_seed is None else abs(s_seed)
    _v_seed: int = randint(0, 2**31 - 1) if v_seed is None else abs(v_seed)
    _logger.info(f"Generating structure with s_seed = {_s_seed}.")
    seed(_s_seed)
    np_seed(_s_seed)
    mock_validator: Validator = deepcopy(validator)
    _strip_check_with(list(mock_validator.schema.values()))  # type: ignore
    _define_coercers(list(mock_validator.schema.values()), mock_validator)  # type: ignore
    schema: dict = {k: _define_structure(deepcopy(v), mock_validator) for k, v in mock_validator.schema.items()}  # type: ignore
    _strip_meta(schema)
    _generate_deps(schema, mock_validator)
    if _LOG_DEBUG:
        _logger.debug(
            f"Pre-customized schema:\n{pformat(schema, indent=4, width=120, sort_dicts=True)}"
        )
    data: list[dict[Any, Any]] = [
        _generate(mock_validator, schema, _v_seed + i, validate) for i in range(num)
    ]
    assert len(data) == num
    return data


def _define_field(definition: dict, require_all: bool) -> None:
    """Determine if a field will be randomly generated.

    Considering the 'required' state, 'read-only' state, whether the validator
    has 'require_all' set and whether all dependencies will be generated.

    If a dependency has yet to be defined (decided to generate or not) then that dependency
    is defined recursively.

    NOTE: If the is a circular dependency in the validator schema this is where it will POP!

    Args
    ----
    definition: The definition of the field.
    require_all: validator.require_all.
    """
    def_req: bool = definition.get("required", False)
    required: bool = def_req or require_all or (not def_req and _required())
    readonly: bool = definition.get("readonly", False) and not (
        "default" in definition or "default_setter" in definition
    )
    define: bool = required and not readonly
    if "dependencies" in definition and define:
        # TODO: _ are the allowed values. Cannot know these at this time.
        # Need a post processing step to check and then remove this dependent field if the
        # criteria are not met.
        if _LOG_DEBUG:
            _logger.debug(f"Definition dependencies: {definition['meta']['deps']}")
        for dep, _ in definition["meta"]["deps"]:
            if dep["meta"].get("defined") is None:
                _define_field(dep, require_all)
            if not dep["meta"]["defined"]:
                define = False
                break
    definition["meta"]["defined"] = define


def _customize_schema(schema: dict, require_all=False) -> dict:
    """Choose which optional fields to populate.

    Args
    ----
    schema: Resolved schema. All registry references have been expanded and dependencies parsed.
    require_all: validator.require_all value.

    Returns
    -------
    schema with optional fields randomly removed.
    """
    for field, definition in tuple(schema.items()):
        if definition["meta"].get("defined") is None:
            _define_field(definition, require_all)
        if not definition["meta"].get("defined"):
            del schema[field]
        typ = definition.get("type")
        if (
            (isinstance(typ, (list, tuple, set)) and "dict" in typ) or typ == "dict"
        ) and ("schema" in definition):
            _customize_schema(definition["schema"], require_all)
    return schema


def _generate(
    validator: Validator, schema: dict, v_seed: int, validate: bool = False
) -> dict:
    """Generate a single record that conforms to validator schema.

    Args
    ----
    validator: A Cerberus Validator
    schema: The resolved schema from the validator.
    validate: User the supplied validator to validate the output.

    Returns
    -------
    A record that conforms to validator schema.
    """
    seed(v_seed)
    np_seed(v_seed)
    _logger.info(f"Generating values with v_seed = {v_seed}.")
    record: dict[Any, Any] = {}
    custom_schema = _customize_schema(deepcopy(schema), validator.require_all)  # type: ignore
    for field, definition in custom_schema.items():
        if definition is not None:
            definition.setdefault("type", TYPES)
            if _LOG_DEBUG:
                _logger.debug(
                    f"Validation rules for '{field}':\n{pformat(definition, indent=4, sort_dicts=True)}."
                )
            _type: str = (
                definition["type"]
                if not isinstance(definition["type"], list)
                else choice(definition["type"])
            )
            record[field] = TYPE_GENERATION[_type](definition)
            if _LOG_DEBUG:
                _logger.debug(
                    f"Generated value for field '{field}':\n{pformat(record[field], indent=4, sort_dicts=True)}."
                )
        else:
            _logger.debug(f"'{field}' not required to be defined or readonly.")

    if validate:
        result: bool = validator.validate(record)  # type: ignore
        if not result:
            message: str = f"Generated data failed validation! {validator.error_str()}. Report this bug!"  # type: ignore
            _logger.error(message)
            _logger.error(f"Generated data:\n{pformat(record, indent=4, sort_dicts=True)}.")
        assert result

    return record
