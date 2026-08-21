#!/usr/bin/env bash
set -euo pipefail

readonly ERROR_CODE="LARA-QUALITY-002"
readonly DEFAULT_CYCLE_SECONDS=120
readonly DEFAULT_CONNECTIONS=16
readonly DEFAULT_SPLIT_SIZE="4M"
readonly RETRY_DELAY_SECONDS=2

if [[ $# -ne 3 ]]; then
  echo "[$ERROR_CODE] Usage: $0 MANIFEST_TSV DOWNLOAD_DIR LOG_DIR" >&2
  exit 2
fi

readonly manifest_path="$1"
readonly download_dir="$2"
readonly log_dir="$3"
readonly cycle_seconds="${LARA_DOWNLOAD_CYCLE_SECONDS:-$DEFAULT_CYCLE_SECONDS}"
readonly connections="${LARA_DOWNLOAD_CONNECTIONS:-$DEFAULT_CONNECTIONS}"
readonly split_size="${LARA_DOWNLOAD_SPLIT_SIZE:-$DEFAULT_SPLIT_SIZE}"

if [[ ! -f "$manifest_path" ]]; then
  echo "[$ERROR_CODE] Download manifest not found: $manifest_path" >&2
  exit 2
fi
if ! command -v aria2c >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
  echo "[$ERROR_CODE] aria2c and sha256sum are required." >&2
  exit 2
fi
if ! [[ "$cycle_seconds" =~ ^[1-9][0-9]*$ && "$connections" =~ ^[1-9][0-9]*$ ]]; then
  echo "[$ERROR_CODE] Cycle seconds and connection count must be positive integers." >&2
  exit 2
fi

mkdir -p "$download_dir" "$log_dir"

download_one() {
  local label="$1"
  local expected_sha256="$2"
  local expected_size="$3"
  local url="$4"
  local output_path="$download_dir/$expected_sha256"
  local log_path="$log_dir/$label.log"
  local command_status
  local actual_size

  while [[ -f "${output_path}.aria2" || ! -f "$output_path" ]]; do
    set +e
    timeout --signal=TERM "$cycle_seconds" aria2c \
      --max-connection-per-server="$connections" \
      --split="$connections" \
      --min-split-size="$split_size" \
      --file-allocation=none \
      --continue=true \
      --check-integrity=true \
      --max-tries=0 \
      --retry-wait="$RETRY_DELAY_SECONDS" \
      --timeout=30 \
      --summary-interval=30 \
      --dir="$download_dir" \
      --out="$expected_sha256" \
      "$url" >>"$log_path" 2>&1
    command_status=$?
    set -e
    if [[ $command_status -ne 0 && $command_status -ne 124 && $command_status -ne 143 ]]; then
      echo "[$ERROR_CODE] $label transfer cycle exited with status $command_status; retrying." >&2
    fi
    sleep "$RETRY_DELAY_SECONDS"
  done

  actual_size="$(stat -c %s "$output_path")"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "[$ERROR_CODE] $label size mismatch: expected=$expected_size actual=$actual_size" >&2
    exit 2
  fi
  printf '%s  %s\n' "$expected_sha256" "$output_path" | sha256sum -c -
}

while IFS=$'\t' read -r label expected_sha256 expected_size url; do
  [[ -z "$label" || "$label" == \#* ]] && continue
  if [[ -z "$expected_sha256" || -z "$expected_size" || -z "$url" ]]; then
    echo "[$ERROR_CODE] Invalid manifest row for $label." >&2
    exit 2
  fi
  download_one "$label" "$expected_sha256" "$expected_size" "$url"
done <"$manifest_path"

echo "ALL_SHARDS_VERIFIED"
