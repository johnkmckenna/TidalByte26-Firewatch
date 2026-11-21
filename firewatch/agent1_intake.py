"""
Agent 1 - Intake & Situation agent (Nemotron).

Job: get trustworthy inputs for the day and hand Agent 2 a grounded situation package.
Nemotron drives the work through tools: it checks which weather days exist, refreshes data
when needed (live mode), runs the risk model, pulls satellite hotspots, inspects suspicious or
important cells, and decides what goes in the handoff. If the requested date is not available
it falls back to the latest available day and says so.

Trust: every number in the handoff is checked against tool outputs by a deterministic
grounding guard; anything not found is dropped and any mismatch is corrected and logged.

Usage
    python -m firewatch.agent1_intake --date 2023-07-03            # replay a past day
    python -m firewatch.agent1_intake --live                        # today
    python -m firewatch.agent1_intake --date 2023-07-03 --offline   # no LLM (fallback/demo safety)
Outputs: runs/<forecast_date>/{risk_grid.parquet, agent1_handoff.json, agent1_trace.json}
"""
import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .featurize import DataNotAvailable
from .live_data import fetch_firms, find_new_ignitions, refresh_gridmet_year
from .nemotron import NemotronClient, NemotronError, run_tool_loop
from .risk_engine import RiskEngine

SYSTEM_PROMPT = """You are Agent 1 (Intake & Situation) in FireWatch, a wildfire early-warning system for {state}.
Your output goes to Agent 2, which assigns danger levels and writes guidance for fire authorities.

Your job, using the tools:
1. Call get_data_status first. Decide which weather date to use (the forecast is for the NEXT day).
   - Replay mode: use the requested date. Live mode: if get_data_status lists years_needing_refresh, call
     refresh_weather for EACH of those years (the current year is usually missing from the cache), then call
     get_data_status again and use the new latest_weather_date.
   - If a date is unavailable, use the latest available date and record that in data_quality.issues.
2. Call run_forecast for that weather date.
3. Call get_hotspots for the forecast date's satellite detections (the day after the weather date in replay;
   today in live mode). If it errors (e.g. no FIRMS key), continue without hotspots and note it.
4. Inspect with get_cell_detail: every hotspot flagged new_ignition, and any top cell whose drivers look
   physically implausible. Check data_quality from the forecast; report warnings.
5. Select up to {n_priority} priority_zones from the zones run_forecast returns. A zone is a group of adjacent
   high-risk cells that dispatchers would treat as one area to watch. Each zone says why it was listed
   (listed_for): "most ignitions expected", "unusual for this place today" (risk_vs_normal = today vs this
   place's normal), or "highest large-fire potential". Keep a mix of all three reasons: busy areas matter for
   staffing, but unusual and large-fire zones are where today is different. Normally keep the list order,
   but promote a zone whose expected_large_fires is clearly higher than the zones ranked above it
   (large fires are rare, so use expected_large_fires, not p_escape_if_ignited, which is a noisy ratio). Identify zones by zone_id exactly as returned.
   Write each reason in one short sentence from the zone's own "conditions" and "baseline" drivers.

Rules:
- Use ONLY numbers returned by tools. Never estimate, round creatively, or invent values or places.
- Identify zones by zone_id and cells by lat_idx/lon_idx exactly as tools return them.
- Do not name towns or landmarks; Agent 2 adds geography.
- notes_for_agent2 is a factual briefing, not advice: do not tell dispatchers what to do, do not guess why a
  data source failed (quote the tool's error text instead), and do not describe weather as "extreme" or "severe";
  "Extreme" is only the name of a risk tier (the top 1% of cells today), not a statement about the weather.
- Be concise and factual in reasons. Use only the drivers listed for that zone, keep each driver's meaning
  (for example, "Fire history in this cell" means fires of any cause, not natural fires). Say when data is missing.

When finished, reply with ONLY this JSON (no prose):
{{
 "mode": "replay" | "live",
 "weather_date": "YYYY-MM-DD",
 "forecast_date": "YYYY-MM-DD",
 "data_quality": {{"status": "ok" | "warning" | "degraded", "issues": ["..."]}},
 "statewide": {{"expected_ignitions": number, "climatology_expected_ignitions": number,
               "ratio_vs_normal": number, "headline": "one sentence"}},
 "priority_zones": [{{"zone_id": "Z1", "reason": "short, cites drivers"}}],
 "hotspot_alerts": [{{"lat_idx": int, "lon_idx": int, "new_ignition": bool, "priority": "P1"|"P2"|"P3",
                     "reason": "short"}}],
 "notes_for_agent2": "2-4 sentences: what matters today and why"
}}"""


def _nums(text):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def reason_problem(reason, z):
    """Return why a free-text reason is not supported by the zone's own drivers, or None if it is."""
    txt = reason.lower()
    drivers = list(z.get("conditions", [])) + list(z.get("baseline", []))
    feats = {d["feature"] for d in drivers}
    if ("natural" in txt or "lightning" in txt) and "cell_prior_natural" not in feats:
        return "mentions natural/lightning fires, which are not among this zone's drivers"
    if "human" in txt and "cell_prior_human" not in feats:
        return "mentions human-caused history, which is not among this zone's drivers"
    allowed = []
    for d in drivers:
        allowed += _nums(str(d["value"])) + _nums(str(d["label"]))
    for k in ("n_cells", "area_km2", "expected_ignitions", "expected_large_fires", "p_ignition_max",
              "risk_vs_normal", "normal_expected_ignitions", "mean_p_ignition"):
        if z.get(k) is not None:
            allowed.append(float(z[k]))
    allowed += [4.0, 5.0]       # "Jul 04" style day numbers and the like are already covered by driver values
    for x in _nums(reason):
        if not any(abs(x - a) <= max(0.051, 0.01 * abs(a)) for a in allowed):
            return f"cites the number {x:g}, which is not in this zone's data"
    return None


class Agent1:
    def __init__(self, engine=None, mode="replay", n_priority=8, firms_fetcher=None, allow_refresh=True):
        self.engine = engine or RiskEngine()
        self.meta = self.engine.builder.meta
        self.mode = mode
        self.n_priority = n_priority
        self.firms_fetcher = firms_fetcher
        self.allow_refresh = allow_refresh and mode == "live"
        self.fc = None
        self.hotspots = None
        self.zones_all = []
        self.zone_of = {}         # (lat_idx, lon_idx) -> zone_id
        self.tool_log = {}        # tool name -> last result (for the grounding guard)

    # ------------------------------------------------------------------ tools
    def get_data_status(self):
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        years = sorted({int(p.stem.split("_")[-1]) for p in Path(self.engine.builder.cache).glob("erc_*.npz")})
        latest = None
        for y in reversed(years):
            latest = self.engine.builder.available_range(y)
            if latest is not None:
                break
        out = {
            "mode": self.mode, "today_utc": str(today.date()),
            "weather_years_cached": f"{years[0]}-{years[-1]}" if years else "none",
            "latest_weather_date": str(latest.date()) if latest is not None else None,
            "weather_lag_days": int((today - latest).days) if latest is not None else None,
            "fire_records_end": self.meta["last_label_date"],
            "models_loaded": sorted(self.engine.models),
            "firms_key_present": bool(config.FIRMS_MAP_KEY) or self.firms_fetcher is not None,
            "refresh_allowed": self.allow_refresh,
        }
        if self.mode == "live":
            need = sorted({today.year} | ({today.year - 1} if today.month == 1 else set()))
            out["years_needing_refresh"] = [y for y in need if y not in years or
                                            (out["weather_lag_days"] or 99) > 2]
        self.tool_log["get_data_status"] = out
        return out

    def refresh_weather(self, year):
        if not self.allow_refresh:
            return {"skipped": True, "reason": "refresh is only allowed in live mode"}
        r = refresh_gridmet_year(int(year), self.engine.builder.cache, config.GRID_META)
        self.tool_log["refresh_weather"] = r
        return r

    def run_forecast(self, weather_date):
        try:
            self.fc = self.engine.forecast(weather_date)
        except DataNotAvailable as e:
            return {"error": "data_not_available", "detail": str(e)}
        self.zones_all = self.engine.watch_zones(self.fc)
        self.zone_of = {tuple(m): z["zone_id"] for z in self.zones_all for m in z["members"]}
        top = []
        for z in self.zones_all[: self.n_priority + 4]:      # compact view for the model (no member list)
            c = {k: v for k, v in z.items() if k not in ("members", "conditions", "baseline", "bbox", "rank_score",
                                                          "mean_p_large_fire", "view_rank_id", "cutoff_percentile")}
            c["conditions"] = [f"{d['label']} {d['value']} ({d['effect']})".replace(" ()", "")
                               for d in z["conditions"]]
            c["baseline"] = [f"{d['label']} {d['value']} ({d['effect']})".replace(" ()", "")
                             for d in z["baseline"]]
            top.append(c)
        # also surface the cells with the highest escape potential among the top 10% of risk
        g = self.fc.grid
        esc = []
        if "p_escape_if_ignited" in g:
            cand = g[(g.risk_percentile >= 0.90) & g.p_escape_if_ignited.notna()]
            cand = cand.sort_values("p_escape_if_ignited", ascending=False).head(5)
            esc = [{"lat_idx": int(r.lat_idx), "lon_idx": int(r.lon_idx), "tier": r.tier,
                    "p_ignition": round(float(r.p_ignition), 5),
                    "p_escape_if_ignited": round(float(r.p_escape_if_ignited), 4)} for r in cand.itertuples()]
        out = {"summary": self.fc.summary, "zones": top, "n_zones_total": len(self.zones_all),
               "high_escape_potential_cells": esc}
        self.tool_log["run_forecast"] = out
        return out

    def get_hotspots(self, date, lookback_days=3):
        if self.fc is None:
            return {"error": "run_forecast first, so hotspots can be matched to cell risk"}
        T = pd.Timestamp(date).normalize()
        lookback_days = int(min(max(lookback_days, 1), 4))
        bbox = self.meta["bbox_south_north_west_east"]
        archive = (pd.Timestamp(datetime.now(timezone.utc).date()) - T).days > 60
        start = T - pd.Timedelta(days=lookback_days)
        try:
            hs, errors = fetch_firms(bbox, day_range=lookback_days + 1, date=start, archive=archive,
                                     fetcher=self.firms_fetcher)
        except Exception as ex:
            out = {"error": f"{type(ex).__name__}: {ex}"}
            self.tool_log["get_hotspots"] = out
            return out
        if len(hs) == 0 and errors and len(errors) == (3 if archive else 4):
            out = {"error": "all FIRMS sources failed", "details": errors}
            self.tool_log["get_hotspots"] = out
            return out
        cand = find_new_ignitions(hs, self.meta, T, lookback_days=lookback_days)
        alerts = []
        g = self.fc.grid.set_index(["lat_idx", "lon_idx"])
        for r in cand.itertuples() if len(cand) else []:
            key = (int(r.lat_idx), int(r.lon_idx))
            if key not in g.index:
                continue      # outside the study box
            cell = g.loc[key]
            pct = float(cell.risk_percentile)
            pesc = None if "p_escape_if_ignited" not in cell or pd.isna(cell.p_escape_if_ignited) \
                else float(cell.p_escape_if_ignited)
            if r.new_ignition and (pct >= 0.90 or (pesc is not None and pesc >= 0.10)):
                sugg = "P1"
            elif r.new_ignition:
                sugg = "P2"
            else:
                sugg = "P3"
            alerts.append({"lat_idx": key[0], "lon_idx": key[1], "lat": round(float(r.latitude), 4),
                           "lon": round(float(r.longitude), 4), "new_ignition": bool(r.new_ignition),
                           "n_detections": int(r.n_detections), "max_frp_mw": None if pd.isna(r.max_frp) else float(r.max_frp),
                           "first_detect_utc": r.first_time, "confidence": r.confidence, "sources": r.sources,
                           "zone_id": self.zone_of.get(key), "cell_tier": cell.tier, "cell_risk_percentile": round(pct, 4),
                           "cell_p_escape_if_ignited": None if pesc is None else round(pesc, 4),
                           "suggested_priority": sugg})
        alerts.sort(key=lambda a: (a["suggested_priority"], -a["cell_risk_percentile"]))
        out = {"date": str(T.date()), "total_detections": int(len(hs)), "cells_with_detections": len(alerts),
               "new_ignition_cells": sum(a["new_ignition"] for a in alerts), "alerts": alerts[:25],
               "source_errors": errors}
        self.hotspots = alerts
        self.tool_log["get_hotspots"] = out
        return out

    def get_cell_detail(self, lat_idx, lon_idx):
        if self.fc is None:
            return {"error": "run_forecast first"}
        rec = self.engine.cell_record(self.fc, int(lat_idx), int(lon_idx))
        if rec is None:
            return {"error": f"cell ({lat_idx}, {lon_idx}) is outside the study area"}
        return rec

    def tools_schema(self):
        def fn(name, desc, props, req):
            return {"type": "function", "function": {"name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": req}}}
        return [
            fn("get_data_status", "Which weather dates are cached, data lag, models loaded, whether FIRMS works.", {}, []),
            fn("refresh_weather", "Live mode only: download the latest gridMET days for a year.",
               {"year": {"type": "integer"}}, ["year"]),
            fn("run_forecast", "Run the ignition-risk model on a weather date (risk is for the next day). "
               "Returns statewide summary, data quality, ranked watch zones (groups of adjacent high-risk "
               "cells) with drivers, and high-escape cells.",
               {"weather_date": {"type": "string", "description": "YYYY-MM-DD"}}, ["weather_date"]),
            fn("get_hotspots", "NASA FIRMS satellite fire detections on a date, flagged new_ignition when no "
               "detections were nearby in the previous days, matched to that cell's risk.",
               {"date": {"type": "string", "description": "YYYY-MM-DD"},
                "lookback_days": {"type": "integer", "description": "1-4, default 3"}}, ["date"]),
            fn("get_cell_detail", "Full risk record and plain-language drivers for one cell.",
               {"lat_idx": {"type": "integer"}, "lon_idx": {"type": "integer"}}, ["lat_idx", "lon_idx"]),
        ]

    def handlers(self):
        return {"get_data_status": self.get_data_status, "refresh_weather": self.refresh_weather,
                "run_forecast": self.run_forecast, "get_hotspots": self.get_hotspots,
                "get_cell_detail": self.get_cell_detail}

    # ------------------------------------------------------------------ grounding guard
    def ground(self, h):
        """Check every cell and number in Nemotron's handoff against tool outputs; fix or drop."""
        report = {"checked": 0, "corrected": [], "dropped": []}
        if self.fc is None:
            raise NemotronError("Agent finished without running a forecast.")
        g = self.fc.grid.set_index(["lat_idx", "lon_idx"])
        s = self.fc.summary
        h.setdefault("statewide", {})
        for k_out, k_tool in [("expected_ignitions", "expected_ignitions_statewide"),
                              ("climatology_expected_ignitions", "climatology_expected_ignitions"),
                              ("ratio_vs_normal", "ratio_vs_normal")]:
            report["checked"] += 1
            if h["statewide"].get(k_out) != s.get(k_tool):
                if h["statewide"].get(k_out) is not None:
                    report["corrected"].append(f"statewide.{k_out}: {h['statewide'].get(k_out)} -> {s.get(k_tool)}")
                h["statewide"][k_out] = s.get(k_tool)
        if s.get("expected_large_fires_statewide") is not None:
            e = float(s["expected_large_fires_statewide"])
            h["statewide"]["expected_large_fires"] = round(e, 3)
            h["statewide"]["chance_of_at_least_one_large_fire"] = round(1 - math.exp(-e), 3)
        if h.get("weather_date") != s["feature_date"] or h.get("forecast_date") != s["forecast_date"]:
            if h.get("weather_date") is not None:
                report["corrected"].append(f"dates -> {s['feature_date']} / {s['forecast_date']}")
            h["weather_date"], h["forecast_date"] = s["feature_date"], s["forecast_date"]

        truth = {z["zone_id"]: z for z in self.zones_all}
        zones, seen = [], set()
        for c in h.get("priority_zones", [])[: self.n_priority]:
            report["checked"] += 1
            zid = str(c.get("zone_id", "")).strip().upper()
            if zid not in truth or zid in seen:
                report["dropped"].append(f"priority zone {c.get('zone_id')!r} not returned by run_forecast or duplicate")
                continue
            seen.add(zid)
            t = truth[zid]
            z = {k: v for k, v in t.items()}          # every number and driver comes from the model, not the LLM
            reason = (c.get("reason") or "").strip()
            problem = reason_problem(reason, t) if reason else "empty reason"
            if problem:
                if reason:
                    report["corrected"].append(f"zone {zid} reason replaced ({problem}): {reason[:90]}")
                reason = self.engine.zone_reason(t)
            z["reason"] = reason
            zones.append(z)
        if not zones and self.zones_all:           # never hand Agent 2 an empty list when zones exist
            report["corrected"].append("no valid priority zones from the agent; used the top-ranked zones")
            zones = [{**t, "reason": self.engine.zone_reason(t)} for t in self.zones_all[: self.n_priority]]
        h["priority_zones"] = zones
        h.pop("priority_cells", None)

        known = {(a["lat_idx"], a["lon_idx"]): a for a in (self.hotspots or [])}
        alerts = []
        for a in h.get("hotspot_alerts", []):
            report["checked"] += 1
            try:
                key = (int(a["lat_idx"]), int(a["lon_idx"]))
            except (KeyError, TypeError, ValueError):
                report["dropped"].append(f"hotspot alert without valid indices: {a}")
                continue
            if key not in known:
                report["dropped"].append(f"hotspot {key} was not returned by get_hotspots")
                continue
            src = known[key]
            if bool(a.get("new_ignition")) != src["new_ignition"]:
                report["corrected"].append(f"hotspot {key}.new_ignition -> {src['new_ignition']}")
            merged = {**src, "priority": a.get("priority", src["suggested_priority"]), "reason": a.get("reason", "")}
            if merged["priority"] not in ("P1", "P2", "P3"):
                merged["priority"] = src["suggested_priority"]
            alerts.append(merged)
        # never let a P1 candidate silently disappear
        for key, src in known.items():
            if src["suggested_priority"] == "P1" and key not in {(x["lat_idx"], x["lon_idx"]) for x in alerts}:
                alerts.append({**src, "priority": "P1", "reason": "Added by grounding guard: new ignition in a high-risk cell."})
                report["corrected"].append(f"hotspot {key} restored as P1")
        h["hotspot_alerts"] = alerts
        dq = h.setdefault("data_quality", {"status": "ok", "issues": []})
        tool_issues = s.get("data_quality", {}).get("issues", [])
        for iss in tool_issues:
            if iss not in dq.get("issues", []):
                dq.setdefault("issues", []).append(iss)
        hs = self.tool_log.get("get_hotspots")
        if any("hotspots" in i for i in dq.get("issues", [])):
            pass                                  # already reported
        elif hs is None:
            tool_issues = tool_issues + ["hotspots not retrieved"]
        elif "error" in hs:
            tool_issues = tool_issues + [f"hotspots unavailable: {hs['error']}"]
        for iss in tool_issues:
            if iss not in dq.get("issues", []):
                dq.setdefault("issues", []).append(iss)
        if tool_issues and dq.get("status") == "ok":
            dq["status"] = "degraded" if any("hotspots" in i for i in tool_issues) else "warning"
        h["grounding_report"] = report
        return h

    # ------------------------------------------------------------------ runs
    def run(self, weather_date=None, client=None, offline=False):
        trace, t0 = [], time.time()
        if self.mode == "replay" and weather_date is None:
            raise ValueError("replay mode needs a weather date")
        if offline:
            h = self._deterministic(weather_date)
            usage = {"calls": 0}
        else:
            client = client or NemotronClient()
            system = SYSTEM_PROMPT.format(state=self.meta.get("state", "the study area"), n_priority=self.n_priority)
            user = (f"Mode: {self.mode}. " + (f"Requested weather date: {weather_date}." if weather_date
                    else "Use the latest available weather date.") + " Produce today's situation package.")
            h, _, usage = run_tool_loop(client, system, user, self.tools_schema(), self.handlers(), trace=trace)
            h["mode"] = self.mode
        h = self.ground(h)
        h["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        h["llm"] = None if offline else {"model": client.model, **usage}
        h["runtime_s"] = round(time.time() - t0, 1)
        out = Path(config.RUNS_DIR) / h["forecast_date"]
        out.mkdir(parents=True, exist_ok=True)
        self.fc.grid.drop(columns=["score"]).to_parquet(out / "risk_grid.parquet", index=False)
        (out / "agent1_handoff.json").write_text(json.dumps(h, indent=2, default=str))
        (out / "agent1_trace.json").write_text(json.dumps(trace, indent=2, default=str))
        return h, out

    def _deterministic(self, weather_date):
        st = self.get_data_status()
        if self.mode == "live" and self.allow_refresh and st.get("years_needing_refresh"):
            for y in st["years_needing_refresh"]:
                print(f"Refreshing gridMET {y} (a few minutes)...")
                self.refresh_weather(y)
            st = self.get_data_status()
        wd = weather_date or st["latest_weather_date"]
        r = self.run_forecast(wd)
        issues = []
        if "error" in r and self.mode == "replay":
            raise DataNotAvailable(r["detail"])
        if "error" in r:
            issues.append(r["detail"])
            r = self.run_forecast(st["latest_weather_date"])
        fdate = self.fc.forecast_date
        hs = self.get_hotspots(str(fdate.date()) if self.mode == "replay" else st["today_utc"]) \
            if st["firms_key_present"] else {"error": "no FIRMS key"}
        if "error" in hs:
            issues.append(f"hotspots unavailable: {hs['error']}")
        s = r["summary"]
        return {
            "mode": self.mode, "weather_date": s["feature_date"], "forecast_date": s["forecast_date"],
            "data_quality": {"status": "ok" if not issues else "degraded", "issues": issues},
            "statewide": {"headline": f"Expected ignitions {s['expected_ignitions_statewide']} "
                                      f"({s['ratio_vs_normal']}x the seasonal norm)."},
            "priority_zones": [{"zone_id": z["zone_id"]} for z in self.zones_all[: self.n_priority]],
            "hotspot_alerts": [{"lat_idx": a["lat_idx"], "lon_idx": a["lon_idx"], "new_ignition": a["new_ignition"],
                                "priority": a["suggested_priority"], "reason": "rule-based"}
                               for a in (self.hotspots or [])],
            "notes_for_agent2": "Deterministic run (no LLM).",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="weather date YYYY-MM-DD (replay); forecast is for the next day")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--offline", action="store_true", help="skip Nemotron; deterministic tool sequence")
    ap.add_argument("--n-priority", type=int, default=8)
    args = ap.parse_args()
    if not args.live and not args.date:
        sys.exit("Give --date YYYY-MM-DD (replay) or --live")
    agent = Agent1(mode="live" if args.live else "replay", n_priority=args.n_priority)
    try:
        h, out = agent.run(args.date, offline=args.offline)
    except NemotronError as e:
        print(f"Nemotron unavailable ({e}); falling back to the deterministic run.")
        h, out = agent.run(args.date, offline=True)
    print(json.dumps({k: h[k] for k in ("weather_date", "forecast_date", "statewide", "data_quality")}, indent=2))
    print(f"priority zones: {len(h['priority_zones'])}, hotspot alerts: {len(h['hotspot_alerts'])}, "
          f"grounding corrections: {len(h['grounding_report']['corrected'])}, dropped: {len(h['grounding_report']['dropped'])}")
    print(f"wrote {out}/agent1_handoff.json")


if __name__ == "__main__":
    main()