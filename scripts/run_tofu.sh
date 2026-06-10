#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/tofu/h2df_lite.yaml}"

h2df-calibrate --config "$CONFIG"
h2df-train --config "$CONFIG"
h2df-evaluate --config "$CONFIG"
