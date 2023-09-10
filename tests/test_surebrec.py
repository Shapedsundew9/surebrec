from logging import NullHandler, getLogger, Logger
from pprint import pformat
from typing import Any

from egp_types.xgc_validator import (
    gGC_entry_validator,
    LGC_entry_validator,
    LGC_json_load_entry_validator,
    LGC_json_dump_entry_validator,
    gms_entry_validator,
)
from pypgtable.validators import (
    database_config_validator,
    raw_table_column_config_validator,
    raw_table_config_validator,
)

from surebrec.surebrec import generate

_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())


def test_LGC_json_load_entry_validator() -> None:
    generate(LGC_json_load_entry_validator, 100, validate=True)


def test_LGC_json_dump_entry_validator() -> None:
    generate(LGC_json_dump_entry_validator, 100, validate=True)


def test_gms_entry_validator() -> None:
    generate(gms_entry_validator, 100, validate=True)


def test_LGC_entry_validator() -> None:
    generate(LGC_entry_validator, 100, validate=True)


def test_GGC_entry_validator() -> None:
    generate(gGC_entry_validator, 100, validate=True)


def test_database_config_validator() -> None:
    generate(database_config_validator, 100, validate=True)


def test_raw_table_column_config_validator() -> None:
    generate(raw_table_column_config_validator, 100, validate=True)


def test_raw_table_config_validator() -> None:
    generate(raw_table_config_validator, 10, validate=True)


def test_random_seed_1() -> None:
    rnd_seed = 7
    a: list[dict[Any, Any]] = generate(raw_table_config_validator, 10, rnd_seed)
    b: list[dict[Any, Any]] = generate(raw_table_config_validator, 10, rnd_seed)
    if a != b:
        for num, (_a, _b) in enumerate(zip(a, b)):
            _logger.debug(
                f"Element {num} does not match:\n"
                f"a:\n{pformat(_a, sort_dicts=True, indent=4)}"
                f"b:\n{pformat(_b, sort_dicts=True, indent=4)}"
            )
            assert _a == _b


def test_random_seed_2() -> None:
    rnd_seed = 7
    a: list[dict[Any, Any]] = generate(raw_table_config_validator, 10, rnd_seed)
    b: list[dict[Any, Any]] = generate(raw_table_config_validator, 10, rnd_seed + 3)
    if a[3:] != b[:-3]:
        for num, (_a, _b) in enumerate(zip(a, b)):
            _logger.debug(
                f"Element {num} does not match:\n"
                f"a:\n{pformat(_a, sort_dicts=True, indent=4)}"
                f"b:\n{pformat(_b, sort_dicts=True, indent=4)}"
            )
            assert _a == _b
