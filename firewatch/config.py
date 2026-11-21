"""Paths and keys. Override any of these with environment variables."""
import os
from pathlib import Path



def _load_dotenv(path):
    """Tiny .env reader (KEY=value lines). Real environment variables win over the file."""
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv(Path(os.environ.get("FIREWATCH_ROOT", ".")).resolve() / ".env")
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT = Path(os.environ.get("FIREWATCH_ROOT", ".")).resolve()
MODEL_DIR = Path(os.environ.get("FIREWATCH_MODEL_DIR", ROOT / "model_out"))
CACHE_DIR = Path(os.environ.get("FIREWATCH_CACHE_DIR", ROOT / "gridmet_cache"))
GRID_META = Path(os.environ.get("FIREWATCH_GRID_META", ROOT / "out" / "grid.json"))
POSITIVES = Path(os.environ.get("FIREWATCH_POSITIVES", ROOT / "out" / "cellday_positives.parquet"))
PLACES = Path(os.environ.get("FIREWATCH_PLACES", ROOT / "out" / "places.csv"))
RUNS_DIR = Path(os.environ.get("FIREWATCH_RUNS_DIR", ROOT / "runs"))

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NEMOTRON_MODEL = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
