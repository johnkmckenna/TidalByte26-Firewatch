#!/usr/bin/env bash
# One-time setup + launch on a Brev CPU instance (run from the repo folder).
set -e
python3 -m pip install --user -r requirements.txt
[ -f .env ] || echo "WARNING: no .env (NVIDIA_API_KEY, FIRMS_MAP_KEY). Precomputed runs still work."
# keep the app running after you disconnect
nohup python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true \
  > streamlit.log 2>&1 &
echo "FireWatch is starting on port 8501 (log: streamlit.log)."
