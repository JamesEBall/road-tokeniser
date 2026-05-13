"""Unit tests for the Safe System rule engine.

Each test is a small, named scenario — easy to point at in a code review.
"""

from __future__ import annotations

import pytest

from road_tokeniser.safe_system import (
    annotate,
    country_default_speed,
    load_rules,
    misalignment_kph,
    priority_score,
    safe_system_speed,
    vru_score,
)


@pytest.fixture(scope="module")
def rules():
    return load_rules()


# ---------------------------------------------------------------------------
# vru_score
# ---------------------------------------------------------------------------


def test_vru_score_zero_for_bare_rural(rules):
    t = {"highway": "trunk"}
    assert vru_score(t, rules) == 0.0


def test_vru_score_school_dominant(rules):
    t = {"highway": "secondary", "school_within_proximity": True}
    assert vru_score(t, rules) == pytest.approx(0.4)


def test_vru_score_residential_plus_crossing(rules):
    t = {
        "highway": "residential",
        "crossing_within_proximity": True,
    }
    # 0.2 (residential) + 0.3 (crossing) = 0.5
    assert vru_score(t, rules) == pytest.approx(0.5)


def test_vru_score_clamped_to_one(rules):
    t = {
        "highway": "residential",
        "school_within_proximity": True,
        "crossing_within_proximity": True,
        "bus_stop_within_proximity": True,
    }
    # 0.4 + 0.3 + 0.2 + 0.1 = 1.0 — exactly the clamp boundary
    assert vru_score(t, rules) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# country_default_speed
# ---------------------------------------------------------------------------


def test_country_default_uk_residential(rules):
    assert country_default_speed("residential", "uk", rules) == 48


def test_country_default_nz_motorway(rules):
    assert country_default_speed("motorway", "nz", rules) == 100


def test_country_default_link_falls_back_to_base(rules):
    assert country_default_speed("motorway_link", "uk", rules) == 113


def test_country_default_unknown_country_uses_generic(rules):
    assert country_default_speed("residential", "atlantis", rules) == 50


# ---------------------------------------------------------------------------
# safe_system_speed — the headline decision tree
# ---------------------------------------------------------------------------


def test_motorway_trusts_posted(rules):
    """Motorways are designed for their posted speed; we trust the design."""
    t = {"highway": "motorway", "posted_speed_kph": 110}
    speed, rule = safe_system_speed(t, rules)
    assert speed == 110
    assert rule == "motorway_passthrough"


def test_school_zone_caps_at_30(rules):
    """A residential street next to a school should be 30 km/h regardless of posted."""
    t = {
        "highway": "residential",
        "posted_speed_kph": 50,
        "vru_score": 0.4,  # school is enough to push us over the threshold
    }
    speed, rule = safe_system_speed(t, rules)
    assert speed == 30
    assert rule == "vru_high"


def test_residential_class_alone_caps_at_30(rules):
    """A residential street with no other VRU markers still caps at 30 — the class itself implies vulnerability."""
    t = {"highway": "residential", "posted_speed_kph": 50, "vru_score": 0.0}
    speed, rule = safe_system_speed(t, rules)
    assert speed == 30
    assert rule == "vru_class"


def test_junction_proximate_caps_at_50(rules):
    """A token within 30 m of a junction faces side-impact risk → 50 km/h cap."""
    t = {
        "highway": "secondary",
        "posted_speed_kph": 80,
        "dist_to_nearest_junction_m": 15.0,
    }
    speed, rule = safe_system_speed(t, rules)
    assert speed == 50
    assert rule == "junction_proximate"


def test_high_curvature_caps_at_60(rules):
    """Winding rural road on a high-speed limit gets curvature-capped."""
    t = {
        "highway": "primary",
        "posted_speed_kph": 100,
        "mean_abs_curvature_rad_per_m": 0.08,
        "dist_to_nearest_junction_m": 500.0,
        "oneway": False,
    }
    speed, rule = safe_system_speed(t, rules)
    assert speed == 60
    assert rule == "high_curvature"


def test_undivided_rural_arterial_caps_at_70(rules):
    """The classic NZ SH1 rural problem: two-way, no median, head-on possible."""
    t = {
        "highway": "trunk",
        "posted_speed_kph": 100,
        "oneway": False,
        "dist_to_nearest_junction_m": 500.0,
        "mean_abs_curvature_rad_per_m": 0.001,
    }
    speed, rule = safe_system_speed(t, rules)
    assert speed == 70
    assert rule == "rural_arterial_undivided"


def test_divided_rural_arterial_allows_90(rules):
    t = {
        "highway": "primary",
        "posted_speed_kph": 100,
        "oneway": True,
        "dist_to_nearest_junction_m": 500.0,
    }
    speed, rule = safe_system_speed(t, rules)
    assert speed == 90
    assert rule == "rural_arterial_divided"


def test_secondary_road_caps_at_60(rules):
    t = {
        "highway": "secondary",
        "posted_speed_kph": 80,
        "dist_to_nearest_junction_m": 500.0,
    }
    speed, rule = safe_system_speed(t, rules)
    assert speed == 60
    assert rule == "secondary_tertiary"


def test_fallthrough_returns_posted(rules):
    """Unrecognised highway class with no other triggers → no opinion."""
    t = {"highway": "cycleway", "posted_speed_kph": 30}
    speed, rule = safe_system_speed(t, rules)
    assert speed == 30
    assert rule == "fallthrough"


# ---------------------------------------------------------------------------
# Downstream: misalignment and priority
# ---------------------------------------------------------------------------


def test_misalignment_positive_when_posted_too_high(rules):
    """The headline case: NZ SH1 100 km/h on an undivided rural section."""
    t = {
        "highway": "trunk",
        "posted_speed_kph": 100,
        "oneway": False,
        "dist_to_nearest_junction_m": 500.0,
    }
    assert misalignment_kph(t, rules) == 30


def test_misalignment_zero_when_safe_system_says_posted(rules):
    t = {"highway": "motorway", "posted_speed_kph": 100}
    assert misalignment_kph(t, rules) == 0


def test_priority_score_higher_with_vru_exposure(rules):
    base = {
        "highway": "secondary",
        "posted_speed_kph": 80,
        "dist_to_nearest_junction_m": 500.0,
    }
    p_low = priority_score(base, rules)
    p_high = priority_score({**base, "vru_score": 0.8}, rules)
    # Same misalignment, but VRU exposure pushes priority higher
    assert p_high > p_low


def test_annotate_populates_all_fields(rules):
    t = {
        "highway": "trunk",
        "posted_speed_kph": 100,
        "oneway": False,
        "dist_to_nearest_junction_m": 500.0,
    }
    out = annotate(t, rules)
    assert out["safe_system_speed_kph"] == 70
    assert out["safe_system_rule"] == "rural_arterial_undivided"
    assert out["misalignment_kph"] == 30
    assert out["priority_score"] > 0
    assert "vru_score" in out
    # Annotate should not mutate the input
    assert "safe_system_speed_kph" not in t
