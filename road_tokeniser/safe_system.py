"""Safe System speed-limit rule engine.

Pure-Python, no I/O, no GIS deps. Loads thresholds from rules/safe_system.yml and
computes a recommended `safe_system_speed_kph` from per-token features.

Usage:
    from road_tokeniser.safe_system import load_rules, safe_system_speed, vru_score

    rules = load_rules()                     # default rules/safe_system.yml
    s = safe_system_speed(token, rules)      # token: dict of features
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "safe_system.yml"


@dataclass(frozen=True)
class Rules:
    """Parsed Safe System policy thresholds."""

    caps: dict[str, int]
    thresholds: dict[str, float]
    vru_weights: dict[str, float]
    country_defaults: dict[str, dict[str, int]]
    version: int

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_RULES_PATH) -> Rules:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            caps=dict(raw["caps"]),
            thresholds={k: float(v) for k, v in raw["thresholds"].items()},
            vru_weights=dict(raw["vru_score_weights"]),
            country_defaults={c: dict(d) for c, d in raw["country_defaults"].items()},
            version=int(raw["version"]),
        )


def load_rules(path: Path | str = DEFAULT_RULES_PATH) -> Rules:
    return Rules.from_yaml(path)


# ---------------------------------------------------------------------------
# Feature derivations
# ---------------------------------------------------------------------------


def vru_score(token: dict[str, Any], rules: Rules) -> float:
    """Vulnerable Road User exposure proxy in [0, 1].

    Components (each contributes if present, weighted, then clamped):
      - school within rules.thresholds.school_proximity_m
      - pedestrian crossing within rules.thresholds.crossing_proximity_m
      - highway class is residential or living_street
      - bus stop within rules.thresholds.bus_stop_proximity_m
    """
    w = rules.vru_weights
    contributions = 0.0

    if token.get("school_within_proximity"):
        contributions += w["school_within_proximity"]
    if token.get("crossing_within_proximity"):
        contributions += w["crossing_within_proximity"]
    if token.get("highway") in {"residential", "living_street"}:
        contributions += w["residential_or_livingstreet_class"]
    if token.get("bus_stop_within_proximity"):
        contributions += w["bus_stop_within_proximity"]

    return max(0.0, min(1.0, contributions))


def country_default_speed(highway: str | None, country: str, rules: Rules) -> int:
    """Posted-speed fallback when OSM `maxspeed` is missing."""
    defaults = rules.country_defaults.get(country, rules.country_defaults["generic"])
    if highway and highway in defaults:
        return int(defaults[highway])
    # Strip the _link suffix and retry: motorway_link -> motorway
    if highway and "_link" in highway:
        base = highway.replace("_link", "")
        if base in defaults:
            return int(defaults[base])
    return int(defaults.get("unclassified", 60))


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------


def safe_system_speed(token: dict[str, Any], rules: Rules) -> tuple[int, str]:
    """Recommended Safe System speed (km/h) + the rule id that fired.

    Decision tree (first match wins):
      1. motorway / motorway_link → trust design (posted speed)
      2. VRU exposure high → pedestrian_mix cap (30)
      3. Junction-proximate → side_impact cap (50)
      4. High curvature on a high-speed road → curvature cap (60)
      5. Undivided high-speed arterial → head_on cap (70)
      6. Divided primary/trunk → divided_default (90)
      7. Secondary/tertiary → 60
      8. Minor / unclassified → 50
      9. Fallback → posted (no opinion)
    """
    caps = rules.caps
    t = rules.thresholds

    highway = token.get("highway")
    posted = int(token.get("posted_speed_kph") or 0)
    oneway = bool(token.get("oneway"))
    junction_dist = float(token.get("dist_to_nearest_junction_m") or 1e9)
    curvature = float(token.get("mean_abs_curvature_rad_per_m") or 0.0)
    score = float(token.get("vru_score") or 0.0)

    # 1. Motorways: trust the design speed
    if highway in {"motorway", "motorway_link"}:
        return posted, "motorway_passthrough"

    # 2. Vulnerable road user exposure → pedestrian mix regime
    if score >= t["vru_score_high"]:
        return caps["pedestrian_mix"], "vru_high"

    if highway in {"residential", "living_street", "service", "pedestrian"}:
        return caps["pedestrian_mix"], "vru_class"

    # 3. Junction proximate → side-impact regime
    if junction_dist < t["junction_proximity_m"]:
        return caps["side_impact"], "junction_proximate"

    # 4. High curvature on a high-speed road
    if curvature >= t["high_curvature_rad_per_m"] and posted >= t["rural_high_speed_kph"]:
        return int(t["high_curvature_cap_kph"]), "high_curvature"

    # 5/6. Undivided vs divided arterial
    if highway in {"trunk", "primary"}:
        # oneway in OSM is a weak proxy for "divided carriageway"; in practice it
        # captures most motorway-class divided arterials (each direction is a
        # separate oneway way). It will miss some divided two-way roads tagged
        # with a separate `divider=*` attribute — Phase B's ML attribute
        # inference fixes this.
        if oneway:
            return int(t["divided_default_kph"]), "rural_arterial_divided"
        return caps["head_on"], "rural_arterial_undivided"

    # 7. Secondary / tertiary
    if highway in {"secondary", "tertiary", "secondary_link", "tertiary_link"}:
        return int(t["secondary_tertiary_kph"]), "secondary_tertiary"

    # 8. Minor / unclassified
    if highway in {"unclassified", "road", "track"}:
        return int(t["minor_road_kph"]), "minor_road"

    # 9. Fall through
    return posted, "fallthrough"


def misalignment_kph(token: dict[str, Any], rules: Rules) -> int:
    """Positive value means posted limit exceeds Safe System recommendation."""
    posted = int(token.get("posted_speed_kph") or 0)
    safe, _ = safe_system_speed(token, rules)
    return posted - safe


def priority_score(token: dict[str, Any], rules: Rules) -> float:
    """Combined intervention-priority score in [0, 1].

    Formula: clamp(misalignment_kph, 0, 50) / 50  *  (0.5 + 0.5 * vru_score)

    The VRU multiplier means a 20-km/h overshoot near a school scores higher
    than the same overshoot on a rural arterial without pedestrian exposure.
    """
    mis = misalignment_kph(token, rules)
    mis_pos = max(0, min(50, mis))
    score = float(token.get("vru_score") or 0.0)
    return (mis_pos / 50.0) * (0.5 + 0.5 * score)


def annotate(token: dict[str, Any], rules: Rules) -> dict[str, Any]:
    """Return token with safe_system_*, misalignment_kph, priority_score added.

    Does not mutate input. Caller is responsible for first computing vru_score
    (we read it from the token) — see vru_score() above. We require the caller
    to have already populated the proximity boolean fields.
    """
    out = dict(token)
    out.setdefault("vru_score", vru_score(out, rules))
    safe, rule_id = safe_system_speed(out, rules)
    out["safe_system_speed_kph"] = safe
    out["safe_system_rule"] = rule_id
    out["misalignment_kph"] = (out.get("posted_speed_kph") or 0) - safe
    out["priority_score"] = priority_score(out, rules)
    return out
