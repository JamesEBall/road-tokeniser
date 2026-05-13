"""Tests for the pure helpers in tokenise.py.

We don't unit-test the GeoPandas pipeline (covered by end-to-end smoke tests),
but parse_maxspeed and _parse_oneway handle the messiest OSM data — they need
exhaustive coverage.
"""

from __future__ import annotations

import math

import pytest

from road_tokeniser.safe_system import load_rules
from road_tokeniser.tokenise import _parse_oneway, parse_maxspeed, utm_epsg_for_bbox


@pytest.fixture(scope="module")
def rules():
    return load_rules()


# ---------------------------------------------------------------------------
# parse_maxspeed
# ---------------------------------------------------------------------------


def test_maxspeed_bare_number_uk_is_mph(rules):
    """UK convention: bare number = mph. 30 mph → ~48 km/h."""
    assert parse_maxspeed("30", "uk", "residential", rules) == 48


def test_maxspeed_bare_number_nz_is_kph(rules):
    """NZ convention: bare number = km/h."""
    assert parse_maxspeed("50", "nz", "residential", rules) == 50


def test_maxspeed_mph_suffix(rules):
    assert parse_maxspeed("60 mph", "uk", "trunk", rules) == 97


def test_maxspeed_kph_suffix(rules):
    assert parse_maxspeed("50 kph", "uk", "residential", rules) == 50
    assert parse_maxspeed("50 km/h", "uk", "residential", rules) == 50


def test_maxspeed_none_falls_back_to_country_default(rules):
    assert parse_maxspeed(None, "uk", "residential", rules) == 48
    assert parse_maxspeed(None, "nz", "motorway", rules) == 100


def test_maxspeed_nan_falls_back(rules):
    assert parse_maxspeed(float("nan"), "nz", "residential", rules) == 50


def test_maxspeed_string_none_walks_signals(rules):
    """OSM uses 'none' for autobahns and 'signals' for variable limits."""
    assert parse_maxspeed("none", "uk", "motorway", rules) == 113
    assert parse_maxspeed("signals", "uk", "motorway", rules) == 113
    assert parse_maxspeed("walk", "uk", "service", rules) == 32


def test_maxspeed_list_picks_smallest(rules):
    """Multi-valued maxspeed → conservative (smallest)."""
    assert parse_maxspeed(["30 mph", "20 mph"], "uk", "residential", rules) == 32


def test_maxspeed_unparseable_falls_back(rules):
    assert parse_maxspeed("abc", "uk", "residential", rules) == 48


def test_maxspeed_link_falls_through_to_base(rules):
    """A `motorway_link` without explicit maxspeed should pick up the motorway default."""
    assert parse_maxspeed(None, "uk", "motorway_link", rules) == 113


# ---------------------------------------------------------------------------
# _parse_oneway — the bug that flipped rural arterials to "divided"
# ---------------------------------------------------------------------------


def test_oneway_list_of_falses_is_false():
    """The bug: bool([False, False]) is True. Helper must unwrap first."""
    assert _parse_oneway([False, False]) is False
    assert _parse_oneway(["no", "no"]) is False


def test_oneway_yes_variants():
    assert _parse_oneway("yes") is True
    assert _parse_oneway("YES") is True
    assert _parse_oneway(True) is True
    assert _parse_oneway("1") is True
    # OSM uses -1 for reverse-direction one-ways; still a one-way for our purposes
    assert _parse_oneway("-1") is True


def test_oneway_no_variants():
    assert _parse_oneway("no") is False
    assert _parse_oneway(None) is False
    assert _parse_oneway(False) is False
    assert _parse_oneway("") is False


# ---------------------------------------------------------------------------
# utm_epsg_for_bbox — used by every metric op in the pipeline
# ---------------------------------------------------------------------------


def test_utm_uk_cambridge():
    """Cambridge UK should be in UTM zone 31N (EPSG 32631)."""
    assert utm_epsg_for_bbox((0.10, 52.18, 0.16, 52.22)) == 32631


def test_utm_nz_wellington():
    """Wellington NZ should be in UTM zone 60S (EPSG 32760)."""
    assert utm_epsg_for_bbox((174.77, -41.30, 174.80, -41.27)) == 32760


def test_utm_southern_hemisphere_picks_327xx():
    epsg = utm_epsg_for_bbox((151.20, -33.86, 151.21, -33.85))  # Sydney
    assert 32701 <= epsg <= 32760
