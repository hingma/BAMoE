#!/usr/bin/env bash
# Download all LTSF benchmark datasets from HuggingFace.
# Files are placed in ./data/ matching the paths expected by run.py.

set -e
DATA_DIR="${1:-./data}"
BASE="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"

mkdir -p "${DATA_DIR}"

download() {
  local url="$1"
  local dest="$2"
  if [ -f "${dest}" ]; then
    echo "  already exists: ${dest}"
  else
    echo "  downloading: $(basename ${dest})"
    curl -L --progress-bar "${url}" -o "${dest}"
  fi
}

echo "=== ETT datasets ==="
download "${BASE}/ETT-small/ETTh1.csv"  "${DATA_DIR}/ETTh1.csv"
download "${BASE}/ETT-small/ETTh2.csv"  "${DATA_DIR}/ETTh2.csv"
download "${BASE}/ETT-small/ETTm1.csv"  "${DATA_DIR}/ETTm1.csv"
download "${BASE}/ETT-small/ETTm2.csv"  "${DATA_DIR}/ETTm2.csv"

echo "=== Weather ==="
download "${BASE}/weather/weather.csv"  "${DATA_DIR}/weather.csv"

echo "=== Traffic ==="
download "${BASE}/traffic/traffic.csv"  "${DATA_DIR}/traffic.csv"

echo "=== Electricity ==="
download "${BASE}/electricity/electricity.csv"  "${DATA_DIR}/electricity.csv"

echo "=== Exchange Rate ==="
download "${BASE}/exchange_rate/exchange_rate.csv"  "${DATA_DIR}/exchange_rate.csv"

echo "=== ILI ====="
download "${BASE}/illness/national_illness.csv"  "${DATA_DIR}/ILI.csv"


echo "=== M4 (Hourly) ==="
mkdir -p "${DATA_DIR}/m4"
download "${BASE}/m4/Hourly-train.csv"  "${DATA_DIR}/m4/Hourly-train.csv"
download "${BASE}/m4/Hourly-test.csv"   "${DATA_DIR}/m4/Hourly-test.csv"

echo ""
echo "All datasets saved to ${DATA_DIR}/"
ls -lh "${DATA_DIR}/"
