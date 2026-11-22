"""
Agent 2 - Danger & Guidance agent (Nemotron).

Input: Agent 1's grounded handoff (runs/<forecast_date>/agent1_handoff.json).
Job: for each watch zone, say WHERE it is (nearest towns), HOW serious it is (watch level), and
WHAT to do (actions chosen from a fixed, vetted action menu), plus a statewide briefing.

Design for trust:
  * Watch levels come from a published rule (LEVEL_RULES). Nemotron may raise a zone by ONE level
    with a stated reason, never lower it. The guard enforces this.
  * Actions must come from the action menu AND be eligible for that zone (e.g. lightning watch only
    if lightning history is a driver; hotspot follow-up only if a detection is in the zone).
    Required actions are added if the model forgot them.
  * Free text is checked: every number must exist in the zone's data, every town named must be one
    the place tool returned for that zone, and banned phrases (evacuation orders, official NWS
    product names) are not allowed. Failing text is replaced by a deterministic summary.

Usage
    python -m firewatch.agent2_danger --date 2024-07-04            # forecast date (Agent 1's output folder)
    python -m firewatch.agent2_danger --date 2024-07-04 --offline  # no LLM
Output: runs/<forecast_date>/{agent2_assessment.json, agent2_trace.json}
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .nemotron import NemotronClient, NemotronError, run_tool_loop
from .places import Places

LEVELS = ["Routine", "Elevated", "High", "Critical"]          # index = severity
LEVEL_RULES = {
    "Critical": "a new-ignition satellite detection (P1/P2) inside the zone, OR expected ignitions in the zone >= 2.0, "
                "OR expected large fires (10+ acres) in the zone >= 0.02",
    "High": "expected ignitions in the zone >= 0.75, OR expected large fires >= 0.005, OR risk >= 3x this "
            "place's normal for the season with expected ignitions >= 0.2",
    "Elevated": "any priority zone (all are in the top 1-3% of cells statewide today)",
    "Routine": "not used for priority zones",
}
STATE_RULES = {
    "Critical": "ratio vs normal >= 3.0 OR chance of at least one large fire >= 0.75",
    "High": "ratio vs normal >= 1.75 OR chance of at least one large fire >= 0.5",
    "Elevated": "ratio vs normal >= 1.2",
    "Routine": "otherwise",
}
BANNED = [r"evacuat", r"red flag warning", r"fire weather watch", r"\bguarantee", r"will (start|ignite|burn)",
          r"certain(ly)? (start|ignite)", r"shelter in place"]


def zone_level(z, hotspots):
    if any(h.get("new_ignition") and h.get("priority") in ("P1", "P2") for h in hotspots):
        return "Critical", "new-ignition satellite detection in zone"
    ei, elf = z.get("expected_ignitions") or 0, z.get("expected_large_fires") or 0
    if ei >= 2.0 or elf >= 0.02:
        return "Critical", f"expected ignitions {ei:.2f}, expected large fires {elf:.4f}"
    if ei >= 0.75 or elf >= 0.005:
        return "High", f"expected ignitions {ei:.2f}, expected large fires {elf:.4f}"
    rvn = z.get("risk_vs_normal") or 0
    if rvn >= 3.0 and ei >= 0.2:
        return "High", f"risk {rvn:.1f}x this place's seasonal normal (expected ignitions {ei:.2f})"
    return "Elevated", "priority zone (top-risk cells today)"


def state_level(sw):
    r, c = sw.get("ratio_vs_normal") or 0, sw.get("chance_of_at_least_one_large_fire") or 0
    if r >= 3.0 or c >= 0.75:
        return "Critical"
    if r >= 1.75 or c >= 0.5:
        return "High"
    if r >= 1.2:
        return "Elevated"
    return "Routine"


def _feats(z, effect="raises risk"):
    return {d["feature"] for d in z.get("conditions", []) + z.get("baseline", []) if d.get("effect") == effect}


# id -> (text template, eligibility(z, ctx) -> bool, required(z, ctx) -> bool)
ACTIONS = {
    "check_detections": ("Review every satellite pass (VIIRS/MODIS) and dispatch/911 smoke reports for this zone.",
                         lambda z, c: True, lambda z, c: True),
    "increase_patrols": ("Increase patrol and lookout coverage in the zone during the afternoon burn period.",
                         lambda z, c: True, lambda z, c: c["level_i"] >= 2),
    "preposition_ia": ("Pre-position initial-attack engines near {towns} to cut response time.",
                       lambda z, c: c["level_i"] >= 2, lambda z, c: c["level_i"] >= 3),
    "notify_local_fd": ("Notify fire departments serving {towns} of today's watch level.",
                        lambda z, c: c["level_i"] >= 2, lambda z, c: False),
    "human_ignition_messaging": ("Push public messaging on human ignition sources (fireworks, campfires, equipment, "
                                 "vehicles) in {towns}.",
                                 lambda z, c: bool({"cell_prior_human", "is_july4_window", "is_holiday_next",
                                                    "is_weekend_next"} & _feats(z)) or c["july4"],
                                 lambda z, c: False),
    "fireworks_patrol": ("Patrol July 4 gathering and fireworks areas near {towns}.",
                         lambda z, c: c["july4"], lambda z, c: False),
    "dry_fuels_brief": ("Brief crews that fuels are dry (drivers: {fuel}); expect faster initial spread.",
                        lambda z, c: bool({"erc", "bi", "fm100", "fm1000", "dry_days", "erc_chg_7d"} & _feats(z)),
                        lambda z, c: False),
    "hot_dry_air_brief": ("Brief crews on hot, dry afternoon air (drivers: {air}).",
                          lambda z, c: bool({"vpd", "vpd_mean_3d", "tmmx", "tmmx_mean_7d", "rmin"} & _feats(z)),
                          lambda z, c: False),
    "wind_watch": ("Watch wind: recent wind is a risk driver here; plan for faster spread.",
                   lambda z, c: bool({"vs", "vs_max_3d"} & _feats(z)), lambda z, c: False),
    "lightning_watch": ("Track lightning strikes over the zone; lightning fire history is a driver here.",
                        lambda z, c: "cell_prior_natural" in _feats(z), lambda z, c: False),
    "large_fire_readiness": ("Confirm extended-attack and aviation availability: this zone's large-fire potential is "
                             "among the highest of today's priority zones.",
                             lambda z, c: (z.get("expected_large_fires") or 0) >= c["elf_median"],
                             lambda z, c: False),
    "verify_hotspot": ("Send a unit or camera check to verify the satellite fire detection in this zone.",
                       lambda z, c: bool(c["hotspots"]), lambda z, c: bool(c["hotspots"])),
}

SYSTEM_PROMPT = """You are Agent 2 (Danger & Guidance) in FireWatch, a wildfire early-warning system for {state}.
Agent 1 has produced today's verified situation: watch zones (groups of high-risk 4-km cells) with model
numbers and drivers. You turn that into a briefing for fire dispatchers.

Use the tools:
1. get_situation - statewide numbers, data quality, and every zone with its rule-based watch level, location
   (towns) and action menu (ids you may choose, required ones marked). This is usually all you need.
2. Optional: find_places(zone_id) for distances/directions, get_action_menu(zone_id, level) if you raise a zone
   (a higher level can unlock more actions and the full action wording), get_level_rules for definitions.
Be efficient: finish in as few tool calls as possible.

For each zone decide:
- level: keep the rule-based level, or raise it by ONE level if you have a concrete reason from the data
  (give raise_reason citing the driver and its value, e.g. "VPD 2.63 kPa with 93°F"). Never lower it.
- actions: 2-5 action ids from that zone's menu, most important first. Include all required ones.
- summary: 1-2 sentences for a dispatcher. Say why the zone is listed (listed_for): a busy area, risk well above
  this place's normal (quote risk_vs_normal, e.g. "3.4x its seasonal normal"), or high large-fire potential;
  where it is (using the towns returned for THAT zone), the level,
  and the main drivers with their values.
Then write a statewide briefing (2-4 sentences) and keep the statewide level from get_situation unless the data
clearly justifies one level higher.

Rules:
- Use only numbers and town names returned by tools, in the zone they belong to. Do not invent places.
- Do not compare zones with superlatives ("highest", "driest of any zone"); state each zone's own values.
- These are forecasts of risk, not predictions that a fire will happen. Do not order evacuations, do not
  name official NWS products (e.g. Red Flag Warning), and do not overstate certainty.
- If data quality is degraded (e.g. no satellite detections), say so once in the briefing.

Reply with ONLY this JSON:
{{"statewide": {{"level": "Routine|Elevated|High|Critical", "briefing": "..."}},
 "zones": [{{"zone_id": "Z1", "level": "...", "raise_reason": null, "actions": ["id", ...], "summary": "..."}}]}}"""


def _nums(text):
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


class Agent2:
    def __init__(self, handoff, places=None):
        self.h = handoff
        self.places = places or Places()
        self.zones = {z["zone_id"]: z for z in handoff.get("priority_zones", [])}
        self.order = [z["zone_id"] for z in handoff.get("priority_zones", [])]
        self.fdate = handoff.get("forecast_date", "")
        self.july4 = self.fdate[5:] in ("07-03", "07-04", "07-05")
        hs = handoff.get("hotspot_alerts", [])
        self.zone_hotspots = {zid: [] for zid in self.zones}
        for a in hs:
            zid = a.get("zone_id")
            if zid in self.zone_hotspots:
                self.zone_hotspots[zid].append(a)
        elfs = sorted(z.get("expected_large_fires") or 0 for z in self.zones.values())
        self.elf_median = elfs[len(elfs) // 2] if elfs else 0
        self.rule = {zid: zone_level(z, self.zone_hotspots[zid]) for zid, z in self.zones.items()}
        self.geo = {zid: self.places.describe_zone(z) for zid, z in self.zones.items()}
        self.tool_log = {}
        self.offline = False

    # ------------------------------------------------------------------ helpers
    def _ctx(self, zid, level=None):
        lvl = level or self.rule[zid][0]
        return {"level_i": LEVELS.index(lvl), "july4": self.july4, "hotspots": self.zone_hotspots[zid],
                "elf_median": self.elf_median}

    def _towns(self, zid, n=2):
        g = self.geo[zid]
        names = [t["name"] for t in g["towns_in_zone"][:n]] or [t["name"] for t in g["nearest_towns"][:n]]
        return " and ".join(names) if names else "the zone"

    def _action_text(self, aid, zid):
        z = self.zones[zid]
        drv = {d["feature"]: f"{d['label'].split(' (')[0].lower()} {d['value']}" for d in z.get("conditions", [])}
        fuel = ", ".join(v for k, v in drv.items() if k in ("erc", "bi", "fm100", "fm1000", "dry_days", "erc_chg_7d"))
        air = ", ".join(v for k, v in drv.items() if k in ("vpd", "vpd_mean_3d", "tmmx", "tmmx_mean_7d", "rmin"))
        return ACTIONS[aid][0].format(towns=self._towns(zid), fuel=fuel or "fuel indices", air=air or "temperature/VPD")

    PRIORITY = ["verify_hotspot", "check_detections", "preposition_ia", "fireworks_patrol", "human_ignition_messaging",
                "large_fire_readiness", "increase_patrols", "wind_watch", "dry_fuels_brief", "lightning_watch",
                "hot_dry_air_brief", "notify_local_fd"]

    def default_actions(self, zid, level, n=4):
        m = {a["id"]: a for a in self.menu(zid, level)}
        acts = [a for a in self.PRIORITY if a in m and m[a]["required"]]
        acts += [a for a in self.PRIORITY if a in m and a not in acts]
        return acts[:max(n, len([a for a in m.values() if a["required"]]))]

    def menu(self, zid, level=None):
        z, c = self.zones[zid], self._ctx(zid, level)
        return [{"id": aid, "text": self._action_text(aid, zid), "required": bool(req(z, c))}
                for aid, (_, elig, req) in ACTIONS.items() if elig(z, c)]

    def _allowed_numbers(self, zid=None):
        sw = self.h.get("statewide", {})
        allowed = [float(v) for v in sw.values() if isinstance(v, (int, float))]
        allowed += [round(float(v) * 100, 1) for k, v in sw.items() if k == "chance_of_at_least_one_large_fire"]
        allowed += [len(self.zones), 1, 2, 3, 4, 5, 10, 24, 7, 2024, 2023, 2022, 2021, 2020]
        allowed += _nums(self.fdate) + _nums(self.h.get("weather_date", ""))
        zids = [zid] if zid else list(self.zones)
        for k in zids:
            z, g = self.zones[k], self.geo[k]
            allowed += _nums(k)
            for d in z.get("conditions", []) + z.get("baseline", []):
                allowed += _nums(str(d.get("value"))) + _nums(d.get("label", ""))
            for key in ("n_cells", "area_km2", "expected_ignitions", "expected_large_fires", "p_ignition_max",
                        "p_escape_if_ignited", "risk_vs_normal", "normal_expected_ignitions"):
                if z.get(key) is not None:
                    allowed.append(float(z[key]))
            for t in g["towns_in_zone"] + g["nearest_towns"] + ([g["nearest_larger_town"]] if g["nearest_larger_town"] else []):
                allowed += [float(t.get("population", 0)), float(t.get("distance_km", 0))]
            allowed += _nums(g["location_label"])
        return allowed

    def _towns_for(self, zid=None):
        zids = [zid] if zid else list(self.zones)
        out = set()
        for k in zids:
            g = self.geo[k]
            out |= {t["name"] for t in g["towns_in_zone"] + g["nearest_towns"]}
            if g["nearest_larger_town"]:
                out.add(g["nearest_larger_town"]["name"])
            out |= set(self.places.mentioned(g["location_label"]))
        return out

    def text_problem(self, text, zid=None):
        low = text.lower()
        for pat in BANNED:
            if re.search(pat, low):
                return f"uses a banned phrase ({pat})"
        allowed_towns = self._towns_for(zid)
        for name in self.places.mentioned(text):
            if name not in allowed_towns:
                return f"names {name}, which is not near {'this zone' if zid else 'any zone'}"
        allowed = self._allowed_numbers(zid)
        for x in _nums(text):
            if not any(abs(x - a) <= max(0.051, 0.015 * abs(a)) for a in allowed):
                return f"cites the number {x:g}, which is not in the data"
        if zid and re.search(r"\b(highest|lowest|driest|hottest|windiest|largest|worst|most|least)\b|of any zone|"
                             r"among (all |the )?zones|than (any|other|all)", low):
            return "makes a comparison with other zones, which the guard cannot verify"
        if "lightning" in low and zid and "cell_prior_natural" not in _feats(self.zones[zid]):
            return "mentions lightning, which is not a driver for this zone"
        return None

    def det_summary(self, zid, level):
        z, g = self.zones[zid], self.geo[zid]
        cond = ", ".join(f"{d['label'].split(' (')[0].lower()} {d['value']}" for d in z.get("conditions", [])
                         if d.get("effect") == "raises risk" and d["feature"] != "time_of_year")[:160]
        s = (f"{zid}, {g['location_label']}: {level} watch level, about {z.get('expected_ignitions', 0):.2f} "
             f"expected ignitions across {z.get('n_cells')} cells")
        if z.get("risk_vs_normal") and "unusual for this place today" in (z.get("listed_for") or []):
            s += f", {z['risk_vs_normal']:.1f}x its seasonal normal"
        return s + (f"; main drivers {cond}." if cond else ".")

    def det_briefing(self, level):
        sw, dq = self.h.get("statewide", {}), self.h.get("data_quality", {})
        crit = [k for k in self.order if self.final_levels.get(k) == "Critical"] if hasattr(self, "final_levels") else []
        s = (f"Statewide watch level {level}: {sw.get('expected_ignitions')} expected ignitions for "
             f"{self.fdate}, {sw.get('ratio_vs_normal')} times the seasonal normal")
        if sw.get("chance_of_at_least_one_large_fire") is not None:
            s += f", with a {round(100 * sw['chance_of_at_least_one_large_fire'])}% chance of at least one fire reaching 10+ acres"
        s += "."
        if crit:
            s += " Critical zones: " + ", ".join(f"{k} ({self.geo[k]['location_label']})" for k in crit) + "."
        if dq.get("status") in ("degraded", "warning") and dq.get("issues"):
            s += " Data note: " + "; ".join(dq["issues"])[:200] + "."
        return s

    # ------------------------------------------------------------------ tools
    def get_situation(self):
        sw = self.h.get("statewide", {})
        zs = []
        for zid in self.order:
            z = self.zones[zid]
            lvl, why = self.rule[zid]
            zs.append({"zone_id": zid, "rule_level": lvl, "rule_basis": why, "tier": z.get("tier"),
                       "n_cells": z.get("n_cells"), "area_km2": z.get("area_km2"),
                       "expected_ignitions": z.get("expected_ignitions"),
                       "expected_large_fires": z.get("expected_large_fires"),
                       "risk_vs_normal": z.get("risk_vs_normal"), "listed_for": z.get("listed_for"),
                       "conditions": [f"{d['label']} {d['value']} ({d['effect']})" for d in z.get("conditions", [])],
                       "baseline": [f"{d['label']} {d['value']} ({d['effect']})" for d in z.get("baseline", [])],
                       "hotspots_in_zone": len(self.zone_hotspots[zid]),
                       "location": {"label": self.geo[zid]["location_label"],
                                    "towns_in_zone": [t["name"] for t in self.geo[zid]["towns_in_zone"][:4]],
                                    "nearest_towns": [f"{t['name']} ({t['distance_km']} km)" for t in
                                                      self.geo[zid]["nearest_towns"][:3]]},
                       "action_menu": [{"id": a["id"], "required": a["required"]} for a in self.menu(zid)]})
        out = {"forecast_date": self.fdate, "statewide": sw, "statewide_rule_level": state_level(sw),
               "data_quality": self.h.get("data_quality"), "zones": zs,
               "agent1_notes": self.h.get("notes_for_agent2", "")}
        self.tool_log["get_situation"] = True
        return out

    def find_places(self, zone_id):
        zid = str(zone_id).strip().upper()
        if zid not in self.zones:
            return {"error": f"unknown zone {zone_id}; valid: {self.order}"}
        return {"zone_id": zid, **self.geo[zid], "place_source": self.places.source}

    def get_action_menu(self, zone_id, level=None):
        zid = str(zone_id).strip().upper()
        if zid not in self.zones:
            return {"error": f"unknown zone {zone_id}; valid: {self.order}"}
        if level not in LEVELS:
            level = None
        return {"zone_id": zid, "for_level": level or self.rule[zid][0], "actions": self.menu(zid, level)}

    def get_level_rules(self):
        return {"zone_levels": LEVEL_RULES, "statewide_levels": STATE_RULES,
                "policy": "Agent 2 may raise a level by one step with a reason, never lower it."}

    def tools_schema(self):
        def fn(name, desc, props, req):
            return {"type": "function", "function": {"name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": req}}}
        z = {"zone_id": {"type": "string", "description": "e.g. Z1"}}
        return [
            fn("get_situation", "Statewide numbers, data quality, zones with drivers and rule-based watch levels.", {}, []),
            fn("find_places", "Towns inside a zone, nearest towns (distance, direction) and a location label.", z, ["zone_id"]),
            fn("get_action_menu", "Actions allowed for a zone (optionally at a raised level) and which are required.",
               {**z, "level": {"type": "string", "enum": LEVELS}}, ["zone_id"]),
            fn("get_level_rules", "Definitions of the watch levels.", {}, []),
        ]

    def handlers(self):
        return {"get_situation": self.get_situation, "find_places": self.find_places,
                "get_action_menu": self.get_action_menu, "get_level_rules": self.get_level_rules}

    # ------------------------------------------------------------------ guard
    def ground(self, out):
        rep = {"checked": 0, "corrected": [], "dropped": []}
        given = {}
        for z in out.get("zones", []) or []:
            zid = str(z.get("zone_id", "")).strip().upper()
            if zid not in self.zones:
                rep["dropped"].append(f"zone {z.get('zone_id')!r} is not one of Agent 1's zones")
                continue
            given.setdefault(zid, z)
        self.final_levels, zones = {}, []
        for zid in self.order:
            rep["checked"] += 1
            src = self.zones[zid]
            rule_lvl, rule_why = self.rule[zid]
            g = given.get(zid)
            if g is None:
                if not self.offline:
                    rep["corrected"].append(f"{zid}: missing from the agent's answer; filled deterministically")
                g = {}
            lvl = g.get("level") if g.get("level") in LEVELS else rule_lvl
            raise_reason = (g.get("raise_reason") or "").strip() or None
            li, ri = LEVELS.index(lvl), LEVELS.index(rule_lvl)
            if li < ri:
                rep["corrected"].append(f"{zid}: level {lvl} is below the rule level {rule_lvl}; restored")
                lvl, raise_reason = rule_lvl, None
            elif li > ri + 1:
                rep["corrected"].append(f"{zid}: level {lvl} is more than one step above {rule_lvl}; capped")
                lvl = LEVELS[ri + 1]
            if LEVELS.index(lvl) > ri:
                prob = self.text_problem(raise_reason, zid) if raise_reason else "no reason given"
                if not prob and not _nums(raise_reason) and not re.search(r"hotspot|detection|satellite", raise_reason.lower()):
                    prob = "reason cites no value from the zone's data"
                if prob:
                    rep["corrected"].append(f"{zid}: raise to {lvl} rejected ({prob}); kept {rule_lvl}")
                    lvl, raise_reason = rule_lvl, None
            else:
                raise_reason = None
            self.final_levels[zid] = lvl
            menu = {a["id"]: a for a in self.menu(zid, lvl)}
            acts = [] if g.get("actions") else self.default_actions(zid, lvl)
            for aid in g.get("actions", []) or []:
                aid = str(aid).strip()
                if aid not in menu:
                    rep["corrected"].append(f"{zid}: action {aid!r} not allowed for this zone; removed")
                elif aid not in acts:
                    acts.append(aid)
            for aid, a in menu.items():
                if a["required"] and aid not in acts:
                    if g:
                        rep["corrected"].append(f"{zid}: required action {aid} added")
                    acts.append(aid)
            if len(acts) < 2:
                for aid in self.default_actions(zid, lvl):
                    if aid not in acts:
                        acts.append(aid)
            summary = (g.get("summary") or "").strip()
            prob = self.text_problem(summary, zid) if summary else "empty"
            if prob:
                if summary and not self.offline:
                    rep["corrected"].append(f"{zid}: summary replaced ({prob}): {summary[:100]}")
                summary = self.det_summary(zid, lvl)
            geo = self.geo[zid]
            zones.append({
                "zone_id": zid, "level": lvl, "rule_level": rule_lvl, "rule_basis": rule_why,
                "raised_by_agent": raise_reason, "location_label": geo["location_label"],
                "towns_in_zone": geo["towns_in_zone"], "nearest_towns": geo["nearest_towns"],
                "nearest_larger_town": geo["nearest_larger_town"], "summary": summary,
                "actions": [{"id": a, "text": menu[a]["text"]} for a in acts[:6]],
                "hotspots": self.zone_hotspots[zid],
                **{k: src.get(k) for k in ("tier", "lat", "lon", "centroid_lat", "centroid_lon", "bbox", "n_cells",
                                           "area_km2", "expected_ignitions", "expected_large_fires",
                                           "p_ignition_max", "conditions", "baseline", "members",
                                           "risk_vs_normal", "normal_expected_ignitions", "listed_for")},
            })
        zones.sort(key=lambda z: (-LEVELS.index(z["level"]), self.order.index(z["zone_id"])))
        sw_in = out.get("statewide", {}) or {}
        rule_sw = state_level(self.h.get("statewide", {}))
        lvl = sw_in.get("level") if sw_in.get("level") in LEVELS else rule_sw
        if LEVELS.index(lvl) < LEVELS.index(rule_sw) or LEVELS.index(lvl) > LEVELS.index(rule_sw) + 1:
            rep["corrected"].append(f"statewide level {lvl} -> {rule_sw}")
            lvl = rule_sw
        rep["checked"] += 1
        brief = (sw_in.get("briefing") or "").strip()
        prob = self.text_problem(brief) if brief else "empty"
        dq = self.h.get("data_quality", {})
        if not prob and dq.get("status") == "degraded" and not re.search(r"satellite|hotspot|firms|data", brief.lower()):
            prob = "does not mention the degraded data"
        if prob:
            if brief and not self.offline:
                rep["corrected"].append(f"statewide briefing replaced ({prob}): {brief[:100]}")
            brief = self.det_briefing(lvl)
        return {"statewide": {**self.h.get("statewide", {}), "level": lvl, "rule_level": rule_sw, "briefing": brief},
                "zones": zones, "grounding_report": rep}

    # ------------------------------------------------------------------ runs
    def run(self, client=None, offline=False, out_dir=None):
        t0, trace, usage = time.time(), [], {"calls": 0}
        self.offline = offline
        if offline:
            raw = {"statewide": {}, "zones": []}
        else:
            client = client or NemotronClient()
            system = SYSTEM_PROMPT.format(state=self.h.get("state", "Colorado"))
            user = f"Forecast date {self.fdate}. There are {len(self.zones)} zones. Produce the danger and guidance package."
            raw, _, usage = run_tool_loop(client, system, user, self.tools_schema(), self.handlers(), trace=trace,
                                          max_steps=24)
        res = self.ground(raw)
        res.update({"forecast_date": self.fdate, "weather_date": self.h.get("weather_date"),
                    "mode": self.h.get("mode"), "data_quality": self.h.get("data_quality"),
                    "place_source": self.places.source, "level_rules": LEVEL_RULES,
                    "disclaimer": "Risk forecast for decision support; not an official fire-weather product.",
                    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "llm": None if offline else {"model": client.model, **usage},
                    "runtime_s": round(time.time() - t0, 1)})
        out = Path(out_dir) if out_dir else Path(config.RUNS_DIR) / self.fdate
        out.mkdir(parents=True, exist_ok=True)
        (out / "agent2_assessment.json").write_text(json.dumps(res, indent=2, default=str, ensure_ascii=False))
        (out / "agent2_trace.json").write_text(json.dumps(trace, indent=2, default=str))
        return res, out


def load_handoff(date=None, path=None):
    p = Path(path) if path else Path(config.RUNS_DIR) / date / "agent1_handoff.json"
    if not p.exists():
        sys.exit(f"No Agent 1 handoff at {p}. Run: python3 -m firewatch.agent1_intake --date <weather date>")
    return json.loads(p.read_text()), p.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="FORECAST date YYYY-MM-DD (the runs/<date> folder Agent 1 wrote)")
    ap.add_argument("--handoff", help="path to an agent1_handoff.json instead of --date")
    ap.add_argument("--offline", action="store_true", help="skip Nemotron; rule-based levels and actions")
    args = ap.parse_args()
    if not args.date and not args.handoff:
        sys.exit("Give --date YYYY-MM-DD (forecast date) or --handoff path")
    h, folder = load_handoff(args.date, args.handoff)
    agent = Agent2(h)
    try:
        res, out = agent.run(offline=args.offline, out_dir=folder)
    except NemotronError as e:
        print(f"Nemotron unavailable ({e}); falling back to the deterministic run.")
        res, out = agent.run(offline=True, out_dir=folder)
    print(f"place source: {res['place_source']}")
    print(f"STATEWIDE {res['statewide']['level']}: {res['statewide']['briefing']}\n")
    for z in res["zones"]:
        print(f"[{z['level']:<8}] {z['zone_id']:<4} {z['location_label']}")
        print(f"           {z['summary']}")
        for a in z["actions"]:
            print(f"             - {a['text']}")
    gr = res["grounding_report"]
    print(f"\ngrounding: {len(gr['corrected'])} corrections, {len(gr['dropped'])} dropped -> {out}/agent2_assessment.json")


if __name__ == "__main__":
    main()
