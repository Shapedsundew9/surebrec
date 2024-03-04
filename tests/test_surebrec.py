"""Test the surebrec module."""
from logging import Logger, NullHandler, getLogger
from pprint import pformat
from typing import Any

import pytest
from egp_types.graph_validators import graph_validator, limited_igraph_validator
from egp_types.xgc_validator import (
    LGC_entry_validator,
    LGC_json_dump_entry_validator,
    LGC_json_load_entry_validator,
    gGC_entry_validator,
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


# Number of batches of test cases to run.
# Some tests generate 10 samples per batch others, that are slower, generate 1.
NUM_TEST_BATCHES: int = 10


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_lgc_json_load_entry_validator(_) -> None:
    """Test that the LGC_json_load_entry_validator works as expected."""
    generate(LGC_json_load_entry_validator, 10, validate=True)


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_lgc_json_dump_entry_validator(_) -> None:
    """Test that the LGC_json_dump_entry_validator works as expected."""
    generate(LGC_json_dump_entry_validator, 10, validate=True)


@pytest.mark.parametrize(
    "s_seed, v_seed", [(1567657329, 1685125724), (1685125724, 1685125724)]
)
def test_lgc_json_dump_entry_validator_nasty_seeds(s_seed, v_seed) -> None:
    """Test that the LGC_json_dump_entry_validator works as expected in some previously failed cases."""
    generate(
        LGC_json_dump_entry_validator, 1, s_seed=s_seed, v_seed=v_seed, validate=True
    )


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_gms_entry_validator(_) -> None:
    """Test that the gms_entry_validator works as expected."""
    generate(gms_entry_validator, 10, validate=True)


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_lgc_entry_validator(_) -> None:
    """Test that the LGC_entry_validator works as expected."""
    generate(LGC_entry_validator, 10, validate=True)


@pytest.mark.parametrize("s_seed, v_seed", [(660482658, 319883203)])
def test_lgc_entry_validator_nasty_seeds(s_seed, v_seed) -> None:
    """Test that the LGC_entry_validator works as expected in some previously failed cases."""
    generate(
        LGC_json_dump_entry_validator, 1, s_seed=s_seed, v_seed=v_seed, validate=True
    )


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_ggc_entry_validator(_) -> None:
    """Test that the gGC_entry_validator works as expected."""
    generate(gGC_entry_validator, 10, validate=True)


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_graph_validator(_) -> None:
    """Test that the graph_validator works as expected."""
    generate(graph_validator, 10, validate=True)


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_igraph_validator(_) -> None:
    """Test that the igraph_validator works as expected."""
    generate(limited_igraph_validator, 1, validate=True)


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_database_config_validator(_) -> None:
    """Test that the database_config_validator works as expected."""
    generate(database_config_validator, 10, validate=True)


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_raw_table_column_config_validator(_) -> None:
    """Test that the raw_table_column_config_validator works as expected."""
    generate(raw_table_column_config_validator, 10, validate=True)


@pytest.mark.parametrize("_", range(NUM_TEST_BATCHES))
def test_raw_table_config_validator(_) -> None:
    """Test that the raw_table_config_validator works as expected."""
    generate(raw_table_config_validator, 1, validate=True)


def test_random_seed_1() -> None:
    """Test that the random seed works as expected."""
    s_seed = 7
    v_seed = 8
    a: list[dict[Any, Any]] = generate(raw_table_config_validator, 10, s_seed, v_seed)
    b: list[dict[Any, Any]] = generate(raw_table_config_validator, 10, s_seed, v_seed)
    if a != b:
        for num, (_a, _b) in enumerate(zip(a, b)):
            _logger.debug(
                f"Element {num} does not match:\n"
                f"a:\n{pformat(_a, sort_dicts=True, indent=4)}"
                f"b:\n{pformat(_b, sort_dicts=True, indent=4)}"
            )
            assert _a == _b


def test_random_seed_2() -> None:
    """Test that the random seed works as expected in the negative case."""
    s_seed = 7
    v_seed = 8
    a: list[dict[Any, Any]] = generate(raw_table_config_validator, 10, s_seed, v_seed)
    b: list[dict[Any, Any]] = generate(
        raw_table_config_validator, 10, s_seed, v_seed + 3
    )
    if a[3:] != b[:-3]:
        for num, (_a, _b) in enumerate(zip(a, b)):
            _logger.debug(
                f"Element {num} does not match:\n"
                f"a:\n{pformat(_a, sort_dicts=True, indent=4)}"
                f"b:\n{pformat(_b, sort_dicts=True, indent=4)}"
            )
            assert _a == _b
