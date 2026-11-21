"""
Agent 3 - Dispatch Communications agent (Nemotron) with human approval.

Input : runs/<date>/agent2_assessment.json (+ agent1_handoff.json for satellite hotspot alerts)
Output: draft messages in runs/<date>/outbox/ with status "pending_approval":
          * one daily monitoring bulletin (all zones, levels, actions)
          * one incident report per new-ignition satellite detection (P1/P2), max 5
        Nothing is "sent" until a person approves it. Approved messages are copied to the mock
        dispatch inbox (runs/dispatch_inbox/) and every decision is written to an audit log.

Nemotron writes the prose (subject, overview, per-zone notes, incident narratives). Facts tables,
action lists, recipients and priority are assembled deterministically from the verified data, and
every sentence Nemotron writes is checked (numbers, town names, banned phrases). Nemotron can call
check_draft to test its text against the guard before submitting - it fixes its own mistakes.

Usage
    python -m firewatch.agent3_dispatch draft   --date 2024-07-04 [--offline]
    python -m firewatch.agent3_dispatch list    --date 2024-07-04
    python -m firewatch.agent3_dispatch show    --date 2024-07-04 --id B-20240704
    python -m firewatch.agent3_dispatch approve --date 2024-07-04 --id B-20240704 --by "J. McKenna"
    python -m firewatch.agent3_dispatch reject  --date 2024-07-04 --id I-20240704-1 --by "J. McKenna" --reason "duplicate"
"""
import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .nemotron import NemotronClient, NemotronError, run_tool_loop
from .places import Places

LEVELS = ["Routine", "Elevated", "High", "Critical"]
BANNED = [r"evacuat", r"red flag warning", r"fire weather watch", r"\bguarantee", r"will (start|ignite|burn)",
          r"shelter in place", r"confirmed fire", r"is (definitely|certainly) a fire"]
DISCLAIMER = ("FireWatch is a decision-support forecast, not an official fire-weather product. Satellite detections "
              "are unverified until confirmed on the ground.")

SYSTEM_PROMPT = """You are Agent 3 (Dispatch Communications) in FireWatch, a wildfire early-warning system for {state}.
Agents 1 and 2 have produced verified risk zones, watch levels, towns, and actions. You draft the messages that a
human duty officer will review and approve before they go to dispatch. You do NOT send anything.

Use the tools:
1. get_assessment - statewide numbers, data quality, zones (level, location, drivers, actions).
2. get_hotspots - new satellite fire detections that need an incident report (may be empty).
3. check_draft(text) - test any sentence you wrote against the fact checker. If it reports a problem, rewrite
   and check again. Check everything before your final answer.

Write:
- bulletin.subject: short, starts with the statewide level, includes the forecast date.
- bulletin.overview: 2-4 sentences for dispatchers: statewide level and numbers, which zones need attention and
  where they are, and any data limits (say once if satellite data was unavailable).
- bulletin.zone_notes: one sentence per zone at High or Critical level (zone_id -> sentence) using that zone's
  location, values, and drivers.
- incident_reports: one per hotspot from get_hotspots (use its ref): a 2-3 sentence narrative: what was detected,
  when, where (nearest town returned by the tool), what the risk model says about that cell, and that it needs
  ground verification.
Facts tables, action lists and recipients are added automatically; do not repeat them.

Rules: only numbers and places that tools returned; no evacuation orders; no NWS product names; detections are
"possible new fires" until verified; plain, calm, professional language.

Reply with ONLY this JSON:
{{"bulletin": {{"subject": "...", "overview": "...", "zone_notes": {{"Z1": "..."}}}},
 "incident_reports": [{{"ref": "H1", "subject": "...", "narrative": "..."}}]}}"""


def _nums(text):
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text))]


def _collect_numbers(obj, out, skip=("members", "bbox")):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in skip:
                _collect_numbers(v, out, skip)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, out, skip)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
        if 0 < obj <= 1:
            out.append(round(float(obj) * 100, 1))
    else:
        out.extend(_nums(obj))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Agent3:
    def __init__(self, assessment, handoff=None, places=None):
        self.a = assessment
        self.h = handoff or {}
        self.places = places or Places()
        self.fdate = assessment["forecast_date"]
        self.zones = {z["zone_id"]: z for z in assessment["zones"]}
        self.hotspots = self._hotspots()
        self.allowed_nums = []
        _collect_numbers({k: v for k, v in assessment.items() if k not in ("level_rules",)}, self.allowed_nums)
        _collect_numbers(self.hotspots, self.allowed_nums)
        self.allowed_nums += _nums(self.fdate) + [0.1, 1, 2, 3, 4, 5, 10, 24, 1000]
        for hs in self.hotspots:
            if hs.get("cell_risk_percentile") is not None:
                self.allowed_nums.append(max(0.1, round(100 * (1 - hs["cell_risk_percentile"]), 1)))
        self.allowed_towns = set()
        for z in assessment["zones"]:
            for t in z.get("towns_in_zone", []) + z.get("nearest_towns", []):
                self.allowed_towns.add(t["name"])
            if z.get("nearest_larger_town"):
                self.allowed_towns.add(z["nearest_larger_town"]["name"])
            self.allowed_towns |= set(self.places.mentioned(z.get("location_label", "")))
        for hs in self.hotspots:
            self.allowed_towns |= {t["name"] for t in hs["nearest_towns"]}

    # ------------------------------------------------------------------ data
    def _hotspots(self):
        out = []
        alerts = [a for a in self.h.get("hotspot_alerts", [])
                  if a.get("new_ignition") and a.get("priority", a.get("suggested_priority")) in ("P1", "P2")]
        alerts.sort(key=lambda a: (a.get("priority", "P3"), -(a.get("cell_risk_percentile") or 0)))
        for i, a in enumerate(alerts[:5], 1):
            near = self.places.near(a["lat"], a["lon"], n=2, max_km=80)
            z = self.zones.get(a.get("zone_id"))
            out.append({"ref": f"H{i}", "lat": a["lat"], "lon": a["lon"], "priority": a.get("priority", a.get("suggested_priority")),
                        "first_detect_utc": a.get("first_detect_utc"), "n_detections": a.get("n_detections"),
                        "max_frp_mw": a.get("max_frp_mw"), "confidence": a.get("confidence"), "sources": a.get("sources"),
                        "zone_id": a.get("zone_id"), "zone_level": z["level"] if z else None,
                        "cell_tier": a.get("cell_tier"), "cell_risk_percentile": a.get("cell_risk_percentile"),
                        "nearest_towns": near,
                        "location_label": (f"{near[0]['distance_km']:.0f} km {near[0]['zone_is_to_the']} of {near[0]['name']}"
                                           if near else f"{a['lat']:.3f}, {a['lon']:.3f}")})
        return out

    def priority(self):
        if any(h["priority"] == "P1" for h in self.hotspots) or self.a["statewide"]["level"] == "Critical":
            return "URGENT"
        if self.a["statewide"]["level"] == "High" or any(z["level"] in ("High", "Critical") for z in self.zones.values()):
            return "PRIORITY"
        return "ROUTINE"

    def recipients(self):
        to = ["State Wildfire Dispatch Center (mock)"]
        cc = []
        for z in self.a["zones"]:
            if z["level"] in ("High", "Critical"):
                towns = [t["name"] for t in z.get("towns_in_zone", [])[:1]] or [t["name"] for t in z.get("nearest_towns", [])[:1]]
                cc += [f"Fire department serving {t} (mock, {z['zone_id']})" for t in towns]
        return to, cc

    def check(self, text):
        """Return a list of problems with a piece of text (empty list = passes)."""
        probs, low = [], str(text).lower()
        for pat in BANNED:
            if re.search(pat, low):
                probs.append(f"banned phrase ({pat})")
        for name in self.places.mentioned(str(text)):
            if name not in self.allowed_towns:
                probs.append(f"names {name}, which is not in the data")
        for x in _nums(text):
            if not any(abs(x - a) <= max(0.051, 0.015 * abs(a)) for a in self.allowed_nums):
                probs.append(f"number {x:g} is not in the data")
        for zid in re.findall(r"\bZ\d+\b", str(text)):
            if zid not in self.zones:
                probs.append(f"{zid} is not a zone")
        return probs

    # ------------------------------------------------------------------ tools
    def get_assessment(self):
        sw = self.a["statewide"]
        return {"forecast_date": self.fdate, "statewide": {k: sw.get(k) for k in (
                    "level", "expected_ignitions", "climatology_expected_ignitions", "ratio_vs_normal",
                    "expected_large_fires", "chance_of_at_least_one_large_fire", "briefing")},
                "data_quality": self.a.get("data_quality"),
                "zones": [{"zone_id": z["zone_id"], "level": z["level"], "location": z["location_label"],
                           "towns_in_zone": [t["name"] for t in z.get("towns_in_zone", [])[:4]],
                           "expected_ignitions": z.get("expected_ignitions"),
                           "expected_large_fires": z.get("expected_large_fires"),
                           "drivers": [f"{d['label']} {d['value']} ({d['effect']})" for d in z.get("conditions", [])],
                           "agent2_summary": z.get("summary"),
                           "actions": [a["id"] for a in z.get("actions", [])]} for z in self.a["zones"]],
                "message_priority": self.priority()}

    def get_hotspots(self):
        return {"count": len(self.hotspots), "hotspots": self.hotspots,
                "note": "Only new-ignition detections rated P1/P2 by Agent 1 need incident reports."}

    def check_draft(self, text):
        p = self.check(text)
        return {"ok": not p, "problems": p}

    def tools_schema(self):
        def fn(name, desc, props, req):
            return {"type": "function", "function": {"name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": req}}}
        return [fn("get_assessment", "Verified statewide numbers, zones, levels, locations, actions.", {}, []),
                fn("get_hotspots", "New satellite fire detections needing incident reports.", {}, []),
                fn("check_draft", "Fact-check a draft sentence or paragraph; returns problems to fix.",
                   {"text": {"type": "string"}}, ["text"])]

    def handlers(self):
        return {"get_assessment": self.get_assessment, "get_hotspots": self.get_hotspots, "check_draft": self.check_draft}

    # ------------------------------------------------------------------ deterministic text
    def det_subject(self):
        sw = self.a["statewide"]
        return f"{sw['level']} wildfire watch - Colorado - {self.fdate} - {len(self.zones)} watch zones"

    def det_overview(self):
        sw, dq = self.a["statewide"], self.a.get("data_quality", {})
        hi = [z for z in self.a["zones"] if z["level"] in ("High", "Critical")]
        s = (f"Statewide watch level {sw['level']} for {self.fdate}: {sw.get('expected_ignitions')} expected ignitions, "
             f"{sw.get('ratio_vs_normal')} times the seasonal normal.")
        if hi:
            s += " Zones needing attention: " + "; ".join(f"{z['zone_id']} {z['location_label']} ({z['level']})" for z in hi) + "."
        if dq.get("status") in ("degraded", "warning"):
            s += " Data note: " + "; ".join(dq.get("issues", []))[:160] + "."
        return s

    def det_zone_note(self, z):
        drv = ", ".join(f"{d['label'].split(' (')[0].lower()} {d['value']}" for d in z.get("conditions", [])
                        if d["effect"] == "raises risk" and d["feature"] != "time_of_year")
        return f"{z['location_label']}: about {z['expected_ignitions']:.2f} expected ignitions" + (f"; {drv}." if drv else ".")

    def det_incident(self, hs):
        t = hs["first_detect_utc"] or "the latest pass"
        return (f"Possible new fire detected by satellite ({hs['sources']}) at {t}, {hs['location_label']}. "
                f"The risk model places this cell in the top {max(0.1, round(100 * (1 - (hs['cell_risk_percentile'] or 0)), 1)):g}% "
                f"of cells statewide today. Ground or camera verification is needed.")

    # ------------------------------------------------------------------ rendering
    def render_bulletin(self, subject, overview, notes):
        sw = self.a["statewide"]
        to, cc = self.recipients()
        lines = [f"**To:** {', '.join(to)}", f"**Cc:** {', '.join(cc) or '-'}", f"**Priority:** {self.priority()}",
                 f"**Subject:** {subject}", "", overview, "",
                 "| Statewide | Value |", "|---|---|",
                 f"| Watch level | {sw['level']} |", f"| Expected ignitions | {sw.get('expected_ignitions')} |",
                 f"| Seasonal normal | {sw.get('climatology_expected_ignitions')} |",
                 f"| Ratio vs normal | {sw.get('ratio_vs_normal')}x |"]
        if sw.get("chance_of_at_least_one_large_fire") is not None:
            lines.append(f"| Chance of at least one 10+ acre fire | {round(100 * sw['chance_of_at_least_one_large_fire'])}% |")
        lines += ["", "| Zone | Level | Location | Expected ignitions |", "|---|---|---|---|"]
        lines += [f"| {z['zone_id']} | {z['level']} | {z['location_label']} | {z['expected_ignitions']:.2f} |"
                  for z in self.a["zones"]]
        for z in self.a["zones"]:
            lines += ["", f"### {z['zone_id']} - {z['level']} - {z['location_label']}"]
            if z["zone_id"] in notes:
                lines.append(notes[z["zone_id"]])
            lines += [f"- {a['text']}" for a in z.get("actions", [])]
        lines += ["", f"_{DISCLAIMER}_"]
        return "\n".join(lines)

    def render_incident(self, hs, subject, narrative):
        to, _ = self.recipients()
        rows = [("Detection time (UTC)", hs["first_detect_utc"]), ("Location", hs["location_label"]),
                ("Coordinates", f"{hs['lat']:.4f}, {hs['lon']:.4f}"), ("Satellite sources", hs["sources"]),
                ("Detections", hs["n_detections"]), ("Max fire radiative power (MW)", hs["max_frp_mw"]),
                ("Confidence", hs["confidence"]), ("Agent 1 priority", hs["priority"]),
                ("Watch zone", f"{hs['zone_id']} ({hs['zone_level']})" if hs["zone_id"] else "outside watch zones"),
                ("Cell risk tier / percentile", f"{hs['cell_tier']} / {hs['cell_risk_percentile']}")]
        lines = [f"**To:** {', '.join(to)}", f"**Priority:** {'URGENT' if hs['priority'] == 'P1' else 'PRIORITY'}",
                 f"**Subject:** {subject}", "", narrative, "", "| Field | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
        lines += ["", f"_{DISCLAIMER}_"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ guard + build
    def build(self, raw):
        rep = {"checked": 0, "corrected": []}

        def take(text, fallback, where):
            rep["checked"] += 1
            text = (text or "").strip()
            if not text:
                if not self.offline:
                    rep["corrected"].append(f"{where}: missing; generated from data")
                return fallback
            p = self.check(text)
            if p:
                rep["corrected"].append(f"{where}: replaced ({'; '.join(p[:2])}): {text[:90]}")
                return fallback
            return text

        b = raw.get("bulletin", {}) or {}
        subject = take(b.get("subject"), self.det_subject(), "bulletin.subject")
        if not subject.lower().startswith(self.a["statewide"]["level"].lower()):
            subject = f"{self.a['statewide']['level']} - {subject}"
        overview = take(b.get("overview"), self.det_overview(), "bulletin.overview")
        dq = self.a.get("data_quality", {})
        if dq.get("status") == "degraded" and not re.search(r"satellite|hotspot|firms|data", overview.lower()):
            overview += " Data note: " + "; ".join(dq.get("issues", []))[:160] + "."
            rep["corrected"].append("bulletin.overview: added the data-quality note")
        given = {str(k).upper(): v for k, v in (b.get("zone_notes") or {}).items()}
        notes = {}
        for z in self.a["zones"]:
            if z["level"] in ("High", "Critical") or z["zone_id"] in given:
                notes[z["zone_id"]] = take(given.get(z["zone_id"]), self.det_zone_note(z), f"zone_notes.{z['zone_id']}")
        msgs = []
        to, cc = self.recipients()
        msgs.append({"id": f"B-{self.fdate.replace('-', '')}", "kind": "bulletin", "priority": self.priority(),
                     "to": to, "cc": cc, "subject": subject,
                     "body_md": self.render_bulletin(subject, overview, notes)})
        reports = {str(r.get("ref", "")).upper(): r for r in raw.get("incident_reports", []) or []}
        for hs in self.hotspots:
            r = reports.get(hs["ref"], {})
            subj = take(r.get("subject"), f"Possible new fire {hs['location_label']} - {hs['priority']} - {self.fdate}",
                        f"incident {hs['ref']}.subject")
            narr = take(r.get("narrative"), self.det_incident(hs), f"incident {hs['ref']}.narrative")
            msgs.append({"id": f"I-{self.fdate.replace('-', '')}-{hs['ref'][1:]}", "kind": "incident",
                         "priority": "URGENT" if hs["priority"] == "P1" else "PRIORITY", "to": to, "cc": [],
                         "subject": subj, "body_md": self.render_incident(hs, subj, narr), "hotspot": hs})
        for extra in set(reports) - {h["ref"] for h in self.hotspots}:
            rep["corrected"].append(f"incident report {extra} dropped: no such hotspot")
        return msgs, rep

    def run(self, client=None, offline=False, out_dir=None):
        t0, trace, usage = time.time(), [], {"calls": 0}
        self.offline = offline
        if offline:
            raw = {}
        else:
            client = client or NemotronClient()
            user = (f"Forecast date {self.fdate}: {len(self.zones)} zones, {len(self.hotspots)} new detections. "
                    "Draft the bulletin and incident reports.")
            raw, _, usage = run_tool_loop(client, SYSTEM_PROMPT.format(state="Colorado"), user,
                                          self.tools_schema(), self.handlers(), trace=trace, max_steps=16)
        msgs, rep = self.build(raw)
        out = Path(out_dir) if out_dir else Path(config.RUNS_DIR) / self.fdate
        box = out / "outbox"
        box.mkdir(parents=True, exist_ok=True)
        llm = None if offline else {"model": client.model, **usage}
        for m in msgs:
            prev = box / f"{m['id']}.json"
            if prev.exists() and json.loads(prev.read_text()).get("status") in ("approved", "rejected"):
                continue      # never overwrite a message a human has already decided on
            m.update({"status": "pending_approval", "created_utc": _now(), "forecast_date": self.fdate,
                      "drafted_by": llm["model"] if llm else "deterministic template",
                      "grounding_report": rep, "content_sha256": hashlib.sha256(m["body_md"].encode()).hexdigest()})
            prev.write_text(json.dumps(m, indent=2, default=str, ensure_ascii=False))
        (out / "agent3_trace.json").write_text(json.dumps(trace, indent=2, default=str))
        summary = {"forecast_date": self.fdate, "messages": [m["id"] for m in msgs], "grounding_report": rep,
                   "llm": llm, "runtime_s": round(time.time() - t0, 1), "generated_utc": _now()}
        (out / "agent3_summary.json").write_text(json.dumps(summary, indent=2))
        _audit(box, {"event": "drafted", "ids": [m["id"] for m in msgs], "by": summary["llm"] or "offline"})
        return msgs, rep, out


# ---------------------------------------------------------------------- human approval (CLI + UI share these)
def _box(date):
    return Path(config.RUNS_DIR) / date / "outbox"


def _audit(box, rec):
    rec = {"utc": _now(), **rec}
    with open(Path(box) / "audit.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def list_messages(date):
    box = _box(date)
    return [json.loads(p.read_text()) for p in sorted(box.glob("*.json"))] if box.exists() else []


def get_message(date, msg_id):
    p = _box(date) / f"{msg_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"no message {msg_id} for {date}")
    return json.loads(p.read_text())


def approve_message(date, msg_id, approver, edited_body=None):
    if not approver or not approver.strip():
        raise ValueError("an approver name is required")
    m = get_message(date, msg_id)
    if m["status"] != "pending_approval":
        raise ValueError(f"{msg_id} is already {m['status']}")
    edited = edited_body is not None and edited_body != m["body_md"]
    if edited:
        m["original_body_md"] = m["body_md"]
        m["body_md"] = edited_body
    m.update({"status": "approved", "approved_by": approver.strip(), "approved_utc": _now(), "edited_by_human": edited,
              "content_sha256": hashlib.sha256(m["body_md"].encode()).hexdigest()})
    (_box(date) / f"{msg_id}.json").write_text(json.dumps(m, indent=2, default=str, ensure_ascii=False))
    inbox = Path(config.RUNS_DIR) / "dispatch_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / f"{date}__{msg_id}.md").write_text(
        f"<!-- delivered {m['approved_utc']} | approved by {m['approved_by']} | sha256 {m['content_sha256']} -->\n"
        + m["body_md"], encoding="utf-8")
    _audit(_box(date), {"event": "approved", "id": msg_id, "by": m["approved_by"], "edited": edited,
                        "sha256": m["content_sha256"]})
    return m


def reject_message(date, msg_id, approver, reason):
    if not approver or not reason:
        raise ValueError("approver and reason are required")
    m = get_message(date, msg_id)
    if m["status"] != "pending_approval":
        raise ValueError(f"{msg_id} is already {m['status']}")
    m.update({"status": "rejected", "rejected_by": approver.strip(), "rejected_utc": _now(), "reject_reason": reason})
    (_box(date) / f"{msg_id}.json").write_text(json.dumps(m, indent=2, default=str, ensure_ascii=False))
    _audit(_box(date), {"event": "rejected", "id": msg_id, "by": approver, "reason": reason})
    return m


def draft(date, offline=False):
    run = Path(config.RUNS_DIR) / date
    ap, hp = run / "agent2_assessment.json", run / "agent1_handoff.json"
    if not ap.exists():
        sys.exit(f"No Agent 2 assessment at {ap}. Run: python3 -m firewatch.agent2_danger --date {date}")
    agent = Agent3(json.loads(ap.read_text()), json.loads(hp.read_text()) if hp.exists() else {})
    try:
        return agent.run(offline=offline, out_dir=run)
    except NemotronError as e:
        print(f"Nemotron unavailable ({e}); using deterministic drafts.")
        return agent.run(offline=True, out_dir=run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draft", "list", "show", "approve", "reject"])
    ap.add_argument("--date", required=True, help="forecast date YYYY-MM-DD")
    ap.add_argument("--id")
    ap.add_argument("--by")
    ap.add_argument("--reason")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    if a.cmd == "draft":
        msgs, rep, out = draft(a.date, a.offline)
        for m in msgs:
            print(f"{m['id']:<16} {m['priority']:<9} {m['subject']}")
        print(f"guard: {len(rep['corrected'])} corrections | drafts in {out}/outbox (status: pending_approval)")
        for c in rep["corrected"]:
            print("  *", c)
    elif a.cmd == "list":
        for m in list_messages(a.date):
            print(f"{m['id']:<16} {m['status']:<17} {m['priority']:<9} {m['subject']}")
    elif a.cmd == "show":
        print(get_message(a.date, a.id)["body_md"])
    elif a.cmd == "approve":
        m = approve_message(a.date, a.id, a.by)
        print(f"approved {m['id']} by {m['approved_by']} -> runs/dispatch_inbox/{a.date}__{m['id']}.md")
    elif a.cmd == "reject":
        m = reject_message(a.date, a.id, a.by, a.reason)
        print(f"rejected {m['id']}: {m['reject_reason']}")


if __name__ == "__main__":
    main()
