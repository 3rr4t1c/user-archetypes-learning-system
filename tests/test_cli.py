"""Tests for the diagnostic scripts' command-line parsing.

These exist because of a real failure: --start was documented as taking an ISO date
and was then handed straight to arles.actions.parse_timestamp, which rejects anything
without a time component. `--start 2024-10-12` blew up with a traceback.

The distinction being pinned here is deliberate and worth keeping:

  * parse_timestamp (data rows)      -- strict. A row whose timestamp lost its time
                                        component is a data problem and must raise.
  * parse_cli_datetime (CLI args)    -- lenient. A human typing a bare date means
                                        midnight UTC.

The leniency belongs at the CLI boundary and must never leak into the data parser.
"""

import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_reshare_density.py"
_spec = importlib.util.spec_from_file_location("check_reshare_density", _SCRIPT)
check_reshare_density = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_reshare_density)

parse_cli_datetime = check_reshare_density.parse_cli_datetime
h_index = check_reshare_density.h_index


def test_bare_date_is_midnight_utc():
    """The exact invocation that failed."""
    assert parse_cli_datetime("2024-10-12") == datetime(2024, 10, 12, tzinfo=timezone.utc)


def test_full_timestamp_still_accepted():
    assert parse_cli_datetime("2024-10-12 13:45:01+00:00") == datetime(
        2024, 10, 12, 13, 45, 1, tzinfo=timezone.utc
    )


def test_surrounding_whitespace_tolerated():
    assert parse_cli_datetime("  2024-10-12  ") == datetime(
        2024, 10, 12, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("bad", ["banana", "", "2024-13-45", "10/12/2024", "2024-10"])
def test_invalid_start_raises_argparse_error_not_a_traceback(bad):
    """argparse turns this into a usage message; a raw exception would be a crash."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_cli_datetime(bad)


def test_cli_leniency_does_not_leak_into_the_data_parser():
    """A bare date must still be rejected for a data row."""
    from arles.actions import MalformedActionError, parse_timestamp

    with pytest.raises(MalformedActionError):
        parse_timestamp("2024-10-12")


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([], 0),
        ([1], 1),
        ([1, 1, 1], 1),
        ([5, 4, 3, 2, 1], 3),
        ([10, 8, 5, 4, 3], 4),
        ([1004], 1),  # one hugely reshared post still gives h=1
        ([3, 3, 3], 3),
    ],
)
def test_h_index(counts, expected):
    assert h_index(counts) == expected
