# FireWatch: inference layer + Agent 1

Put the `firewatch/` folder in your repo root, next to `model_out/`, `gridmet_cache/` and `out/`.

    pip install -r requirements.txt
    export NVIDIA_API_KEY=nvapi-...          # build.nvidia.com
    export FIRMS_MAP_KEY=...                 # firms.modaps.eosdis.nasa.gov/api/map_key (free)

    # replay a past day (weather date; forecast is for the next day)
    python3 -m firewatch.agent1_intake --date 2023-07-03
    # live (latest weather; refreshes gridMET if stale)
    python3 -m firewatch.agent1_intake --live
    # no LLM (demo safety net)
    python3 -m firewatch.agent1_intake --date 2023-07-03 --offline

Outputs: runs/<forecast_date>/risk_grid.parquet, agent1_handoff.json, agent1_trace.json
Model: NEMOTRON_MODEL (default nvidia/nemotron-3.5-lightning-30b-a3b). Paths: FIREWATCH_* env vars (see config.py).
