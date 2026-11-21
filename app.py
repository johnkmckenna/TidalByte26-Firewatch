"""
FireWatch dashboard (Streamlit).

    streamlit run app.py

Reads the files the agents write to runs/<forecast_date>/:
  risk_grid.parquet, agent1_handoff.json, agent2_assessment.json, outbox/*.json, agent*_trace.json
Works on precomputed runs alone (hosting / demo). If the model, weather cache and API keys are present,
the sidebar can also run the full pipeline (Agent 1 -> 2 -> 3) for a new date.
"""
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from firewatch import config

st.set_page_config(page_title="FireWatch - wildfire ignition watch", page_icon="🔥", layout="wide")

RUNS = Path(config.RUNS_DIR)
STEP = 1 / 24
LEVEL_COLOR = {"Critical": [190, 30, 45], "High": [235, 110, 30], "Elevated": [240, 190, 40], "Routine": [120, 160, 120]}
LEVEL_EMOJI = {"Critical": "🟥", "High": "🟧", "Elevated": "🟨", "Routine": "🟩"}


# ---------------------------------------------------------------------- data
def list_runs():
    if not RUNS.exists():
        return []
    return sorted([p.name for p in RUNS.iterdir() if (p / "agent1_handoff.json").exists()], reverse=True)


@st.cache_data(show_spinner=False)
def load_json(path, mtime):
    return json.loads(Path(path).read_text())


def jload(p):
    p = Path(p)
    return load_json(str(p), p.stat().st_mtime) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_grid(path, mtime):
    g = pd.read_parquet(path)
    keep = g["risk_percentile"] >= 0.90                     # top 10% of cells per view is enough for the map
    for c in ("unusual_percentile", "large_fire_percentile"):
        if c in g:
            keep |= g[c] >= 0.90
    return g[keep].copy()


def square(lat, lon):
    h = STEP / 2
    return [[lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h], [lon - h, lat + h]]


def pipeline_available():
    need = [Path(config.MODEL_DIR) / "y_any" / "model.json", Path(config.GRID_META), Path(config.CACHE_DIR)]
    return all(p.exists() for p in need)


def run_pipeline(weather_date, offline, log):
    from firewatch.agent1_intake import Agent1
    from firewatch.agent2_danger import Agent2
    from firewatch.agent3_dispatch import draft
    from firewatch.nemotron import NemotronError
    log.write("Agent 1: intake, forecast, satellite hotspots ...")
    a1 = Agent1(mode="replay")
    try:
        h, out = a1.run(weather_date, offline=offline)
    except NemotronError as e:
        log.write(f"Nemotron unavailable ({e}); deterministic Agent 1.")
        h, out = a1.run(weather_date, offline=True)
    log.write(f"Agent 1 done: {len(h['priority_zones'])} zones. Agent 2: danger levels and guidance ...")
    a2 = Agent2(h)
    try:
        a2.run(offline=offline, out_dir=out)
    except NemotronError as e:
        log.write(f"Nemotron unavailable ({e}); deterministic Agent 2.")
        a2.run(offline=True, out_dir=out)
    log.write("Agent 3: drafting dispatch messages ...")
    draft(h["forecast_date"], offline=offline)
    return h["forecast_date"]


# ---------------------------------------------------------------------- sidebar
st.sidebar.title("🔥 FireWatch")
st.sidebar.caption("Next-day wildfire ignition watch for Colorado. Three Nemotron agents, human approval.")
runs = list_runs()
if "fdate" not in st.session_state and runs:
    st.session_state.fdate = runs[0]
if runs:
    st.session_state.fdate = st.sidebar.selectbox("Forecast date", runs,
                                                  index=runs.index(st.session_state.fdate) if st.session_state.get("fdate") in runs else 0)

with st.sidebar.expander("Run the pipeline for a new date", expanded=not runs):
    if pipeline_available():
        wd = st.date_input("Weather date (forecast is for the next day)", value=date(2024, 7, 3),
                           min_value=date(1992, 2, 1), max_value=date.today() - timedelta(days=1))
        offline = st.checkbox("Offline (no Nemotron)", value=not bool(config.NVIDIA_API_KEY))
        if st.button("Run Agents 1 → 2 → 3", type="primary"):
            with st.status("Running agents (Nemotron runs take 2-5 minutes)...", expanded=True) as s:
                try:
                    fd = run_pipeline(str(wd), offline, s)
                    s.update(label=f"Done: forecast for {fd}", state="complete")
                    st.session_state.fdate = fd
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    s.update(label="Pipeline failed", state="error")
                    st.exception(e)
    else:
        st.info("Live runs need model_out/, gridmet_cache/ and out/grid.json. This deployment shows precomputed runs.")

st.sidebar.markdown("---")
st.sidebar.caption("Decision-support forecast, not an official fire-weather product. "
                   "Satellite detections are unverified until confirmed on the ground.")

if not runs:
    st.title("FireWatch")
    st.warning(f"No runs found in {RUNS}. Run `python3 -m firewatch.agent1_intake --date 2024-07-03`, then Agents 2 and 3.")
    st.stop()

fdate = st.session_state.fdate
run = RUNS / fdate
h1 = jload(run / "agent1_handoff.json")
a2 = jload(run / "agent2_assessment.json")
grid_path = run / "risk_grid.parquet"
grid = load_grid(str(grid_path), grid_path.stat().st_mtime) if grid_path.exists() else None

# ---------------------------------------------------------------------- header
sw = (a2 or {}).get("statewide") or h1.get("statewide", {})
level = sw.get("level", "-")
st.title(f"{LEVEL_EMOJI.get(level, '')} Wildfire watch for {fdate}")
c = st.columns(5)
c[0].metric("Statewide level", level)
c[1].metric("Expected ignitions", sw.get("expected_ignitions"))
c[2].metric("vs seasonal normal", f"{sw.get('ratio_vs_normal')}x", help=f"normal: {sw.get('climatology_expected_ignitions')}")
ch = sw.get("chance_of_at_least_one_large_fire")
c[3].metric("Chance of a 10+ acre fire", f"{round(100 * ch)}%" if ch is not None else "-")
c[4].metric("New satellite detections", sum(1 for a in h1.get("hotspot_alerts", []) if a.get("new_ignition")))
dq = (a2 or h1).get("data_quality", {})
if dq.get("status") not in (None, "ok"):
    st.warning("Data quality: " + "; ".join(dq.get("issues", [])))
if a2 and a2["statewide"].get("briefing"):
    st.info(a2["statewide"]["briefing"])
elif not a2:
    st.warning("Agent 2 has not run for this date: `python3 -m firewatch.agent2_danger --date " + fdate + "`")

zones = (a2 or {}).get("zones") or h1.get("priority_zones", [])
tab_map, tab_out, tab_trace, tab_about = st.tabs(["🗺️ Map & zones", "📨 Dispatch outbox", "🔍 Agent trace & grounding", "ℹ️ About the model"])

# ---------------------------------------------------------------------- map
with tab_map:
    views = {"Most ignitions expected": "risk_percentile"}
    if grid is not None and "unusual_percentile" in grid:
        views["Unusual for this place today"] = "unusual_percentile"
    if grid is not None and "large_fire_percentile" in grid:
        views["Large-fire potential"] = "large_fire_percentile"
    view = st.radio("Shade the map by", list(views), horizontal=True,
                    help="Most ignitions: where fires are most likely tomorrow (follows people). Unusual: today's "
                         "risk divided by this place's normal for the season. Large-fire: chance a fire starts and "
                         "reaches 10+ acres.")
    vcol = views[view]
    layers = []
    if grid is not None and len(grid):
        g = grid[grid[vcol] >= 0.90].copy()
        g["polygon"] = [square(a, b) for a, b in zip(g.cell_lat, g.cell_lon)]
        t = ((g[vcol] - 0.90) / 0.10).clip(0, 1)
        base = {"risk_percentile": (255, 60), "unusual_percentile": (150, 230), "large_fire_percentile": (120, 40)}[vcol]
        g["color"] = [[base[0], int(200 - 160 * v), base[1], int(40 + 130 * v)] for v in t]
        extra = g["risk_vs_normal"] if "risk_vs_normal" in g else pd.Series([None] * len(g), index=g.index)
        g["tip"] = [f"P(ignition) {q:.4f} · risk percentile {p:.3f}" + (f" · {r:.1f}x normal" if r == r and r is not None else "")
                    for p, q, r in zip(g.risk_percentile, g.p_ignition, extra)]
        layers.append(pdk.Layer("PolygonLayer", g[["polygon", "color", "tip"]], get_polygon="polygon",
                                get_fill_color="color", stroked=False, pickable=True))
    zrows = []
    for z in zones:
        col = LEVEL_COLOR.get(z.get("level"), [200, 60, 60])
        for la, lo in z.get("members", []):
            lat = 25.066666666666666 + la * STEP
            lon = -124.76666666666667 + lo * STEP
            zrows.append({"polygon": square(lat, lon), "color": col + [170],
                          "tip": f"{z['zone_id']} · {z.get('level', z.get('tier'))} · {z.get('location_label', '')}"})
    if zrows:
        layers.append(pdk.Layer("PolygonLayer", pd.DataFrame(zrows), get_polygon="polygon", get_fill_color="color",
                                get_line_color=[60, 20, 20], line_width_min_pixels=1, stroked=True, pickable=True))
        labels = pd.DataFrame([{"lat": z["centroid_lat"], "lon": z["centroid_lon"], "text": z["zone_id"]} for z in zones])
        layers.append(pdk.Layer("TextLayer", labels, get_position="[lon, lat]", get_text="text", get_size=16,
                                get_color=[20, 20, 20], get_alignment_baseline="'bottom'"))
    hs = [a for a in h1.get("hotspot_alerts", [])]
    if hs:
        hdf = pd.DataFrame([{"lat": a["lat"], "lon": a["lon"],
                             "color": [220, 0, 0] if a.get("new_ignition") else [120, 120, 120],
                             "tip": f"Satellite detection · {'NEW' if a.get('new_ignition') else 'ongoing'} · "
                                    f"{a.get('priority', a.get('suggested_priority'))} · {a.get('first_detect_utc')}"} for a in hs])
        layers.append(pdk.Layer("ScatterplotLayer", hdf, get_position="[lon, lat]", get_fill_color="color",
                                get_radius=3500, radius_min_pixels=5, pickable=True, stroked=True,
                                get_line_color=[255, 255, 255], line_width_min_pixels=1))
    vstate = pdk.ViewState(latitude=39.0, longitude=-105.55, zoom=6.0)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=vstate, tooltip={"text": "{tip}"},
                             map_style="light"), height=560)
    st.caption(f"Shaded: top 10% of 4-km cells by '{view.lower()}'. Outlined: watch zones colored by level "
               "(🟥 Critical 🟧 High 🟨 Elevated). Red dots: new satellite fire detections; grey: ongoing fires.")

    st.subheader("Watch zones")
    for z in zones:
        lvl = z.get("level", z.get("tier"))
        why = " / ".join(z.get("listed_for") or [])
        rvn = f" · {z['risk_vs_normal']:.1f}x normal" if z.get("risk_vs_normal") else ""
        title = f"{LEVEL_EMOJI.get(lvl, '')} {z['zone_id']} · {lvl} · {z.get('location_label', '')} · " \
                f"{z.get('expected_ignitions', 0):.2f} expected ignitions{rvn}" + (f" · {why}" if why else "")
        with st.expander(title, expanded=lvl == "Critical"):
            left, right = st.columns([3, 2])
            with left:
                st.write(z.get("summary") or z.get("reason", ""))
                if z.get("actions"):
                    st.markdown("**Recommended actions**")
                    st.markdown("\n".join(f"- {a['text']}" for a in z["actions"]))
                if z.get("raised_by_agent"):
                    st.markdown(f"*Level raised by Agent 2 from {z.get('rule_level')}:* {z['raised_by_agent']}")
            with right:
                st.markdown("**Today's conditions**")
                st.markdown("\n".join(f"- {d['label']}: {d['value']} ({d['effect']})" for d in z.get("conditions", [])))
                st.markdown("**Baseline**")
                st.markdown("\n".join(f"- {d['label']}: {d['value']}" for d in z.get("baseline", [])))
                towns = [t["name"] for t in z.get("towns_in_zone", [])]
                if towns:
                    st.markdown("**Towns in zone:** " + ", ".join(towns))
                st.caption(f"{z.get('n_cells')} cells · {z.get('area_km2')} km² · expected large fires "
                           f"{z.get('expected_large_fires')} · rule: {z.get('rule_basis', '-')}")

# ---------------------------------------------------------------------- outbox
with tab_out:
    from firewatch.agent3_dispatch import approve_message, draft, list_messages, reject_message
    msgs = list_messages(fdate)
    if not msgs:
        st.info("No drafts yet.")
        if a2 and st.button("Draft dispatch messages (Agent 3)"):
            with st.spinner("Agent 3 drafting..."):
                draft(fdate, offline=not bool(config.NVIDIA_API_KEY))
            st.rerun()
    approver = st.text_input("Duty officer name (required to approve or reject)", key="approver")
    for m in msgs:
        badge = {"pending_approval": "⏳ pending", "approved": "✅ approved", "rejected": "❌ rejected"}[m["status"]]
        with st.expander(f"{badge} · {m['priority']} · {m['subject']}", expanded=m["status"] == "pending_approval"):
            st.caption(f"{m['id']} · drafted by {m.get('drafted_by')} · {m.get('created_utc')}")
            if m["status"] == "pending_approval":
                body = st.text_area("Edit before approving", m["body_md"], height=320, key=f"body_{m['id']}")
                with st.container(border=True):
                    st.markdown(body)
                b1, b2, b3 = st.columns([1, 1, 3])
                reason = b3.text_input("Reason (for reject)", key=f"reason_{m['id']}")
                if b1.button("✅ Approve & send", key=f"ok_{m['id']}", type="primary", disabled=not approver):
                    approve_message(fdate, m["id"], approver, edited_body=body)
                    st.success("Sent to the mock dispatch inbox.")
                    st.rerun()
                if b2.button("❌ Reject", key=f"no_{m['id']}", disabled=not (approver and reason)):
                    reject_message(fdate, m["id"], approver, reason)
                    st.rerun()
            else:
                who = m.get("approved_by") or m.get("rejected_by")
                st.caption(f"{m['status']} by {who}" + (" (edited by human)" if m.get("edited_by_human") else "")
                           + (f" · reason: {m['reject_reason']}" if m.get("reject_reason") else ""))
                st.markdown(m["body_md"])
            gr = m.get("grounding_report", {})
            if gr.get("corrected"):
                st.caption("Guard corrections: " + " | ".join(gr["corrected"]))
    audit = run / "outbox" / "audit.jsonl"
    if audit.exists():
        with st.expander("Audit log"):
            st.dataframe(pd.DataFrame([json.loads(l) for l in audit.read_text().splitlines() if l.strip()]),
                         use_container_width=True)

# ---------------------------------------------------------------------- trace
with tab_trace:
    st.markdown("Every number and place the agents output is checked against tool results by a deterministic "
                "grounding guard. Below: what each Nemotron agent did and what the guard corrected.")
    a3s = jload(run / "agent3_summary.json")
    for name, doc, trace_file in [("Agent 1 · Intake & situation", h1, "agent1_trace.json"),
                                  ("Agent 2 · Danger & guidance", a2, "agent2_trace.json"),
                                  ("Agent 3 · Dispatch drafts", a3s, "agent3_trace.json")]:
        st.subheader(name)
        if not doc:
            st.caption("not run")
            continue
        llm = doc.get("llm")
        k = st.columns(4)
        k[0].metric("Model", (llm or {}).get("model", "offline").split("/")[-1])
        k[1].metric("LLM calls", (llm or {}).get("calls", 0))
        gr = doc.get("grounding_report", {})
        k[2].metric("Guard corrections", len(gr.get("corrected", [])))
        k[3].metric("Runtime (s)", doc.get("runtime_s", "-"))
        for line in gr.get("corrected", []) + gr.get("dropped", []):
            st.markdown(f"- 🛡️ {line}")
        tr = jload(run / trace_file)
        if tr:
            rows = []
            for e in tr:
                if e["type"] == "llm":
                    calls = ", ".join(c["name"] for c in e.get("tool_calls", []))
                    rows.append({"step": e["step"], "event": "Nemotron", "detail": f"calls: {calls}" if calls else "final answer",
                                 "latency_s": e.get("latency_s")})
                elif e["type"] == "tool":
                    rows.append({"step": e["step"], "event": f"tool · {e['name']}",
                                 "detail": json.dumps(e.get("args"))[:120] + ("" if e.get("ok") else "  ⚠ error"),
                                 "latency_s": e.get("latency_s")})
                else:
                    rows.append({"step": e["step"], "event": e["type"], "detail": e.get("error", ""), "latency_s": None})
            with st.expander(f"Tool-call trace ({len(rows)} events)"):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**What FireWatch does.** Every day it forecasts where new wildfires are most likely to start tomorrow across
Colorado (16,128 cells of ~4 km), groups the riskiest cells into watch zones, sets a watch level per zone,
recommends actions, checks NASA FIRMS satellite detections for possible new fires, and drafts dispatch
messages that a duty officer approves before anything is sent.

**Pipeline**
1. **Risk model** - XGBoost trained on 30 years of US Forest Service fire records (FPA-FOD) and daily gridMET
   weather/fuel data; calibrated probabilities; per-cell explanations from exact feature contributions.
2. **Agent 1 (Nemotron)** - checks data, runs the model, pulls satellite hotspots, picks watch zones.
3. **Agent 2 (Nemotron)** - names towns, sets watch levels within published rules, picks vetted actions.
4. **Agent 3 (Nemotron)** - drafts the bulletin and incident reports, fact-checks itself with a tool.
5. **Human approval** - nothing reaches dispatch without a named approver; every decision is audited.
""")
    mpath = Path(config.MODEL_DIR)
    for t, label in [("y_any", "Any new ignition"), ("y_escaped", "Fire reaching 10+ acres")]:
        f = mpath / t / "metrics.csv"
        if f.exists():
            st.markdown(f"**Backtest on held-out years - {label}**")
            df = pd.read_csv(f)
            cols = ["method", "n_test_ignitions"] + [c for c in df.columns if c.startswith("capture@")] + ["roc_auc"]
            st.dataframe(df[[c for c in cols if c in df.columns]].round(3), use_container_width=True, hide_index=True)
    st.caption("capture@k%: share of real ignitions that fell in the top k% of cells flagged that day. "
               "Random flagging captures k%.")
