#!/usr/bin/env bash
# =============================================================================
#  SentinelPay — One-Click Launcher
#  University of Mumbai | BDA Semester 7 Project
# =============================================================================
set -euo pipefail

# ── Terminal styling ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

banner() {
  echo -e "${CYAN}${BOLD}"
  echo "  ╔══════════════════════════════════════════════════════════════╗"
  echo "  ║   🛡️  SentinelPay — Fraud & Anomaly Intelligence Platform   ║"
  echo "  ║       University of Mumbai · BDA Semester 7 · 2024          ║"
  echo "  ╚══════════════════════════════════════════════════════════════╝"
  echo -e "${RESET}"
}

step() { echo -e "${BOLD}${YELLOW}▶ $1${RESET}"; }
ok()   { echo -e "${GREEN}  ✓ $1${RESET}"; }
err()  { echo -e "${RED}  ✗ ERROR: $1${RESET}"; exit 1; }

# ── Determine project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

banner

# ── 1. Python availability check ─────────────────────────────────────────────
step "Checking Python installation…"
PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" --version 2>&1 | awk '{print $2}')
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
      PYTHON="$cmd"
      ok "Found $cmd $VER"
      break
    fi
  fi
done
[ -z "$PYTHON" ] && err "Python 3.9+ is required. Install from https://python.org"

# ── 2. Virtual environment ────────────────────────────────────────────────────
step "Setting up virtual environment (.venv)…"
if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
  ok "Created .venv"
else
  ok ".venv already exists — reusing"
fi

# Activate
if [ -f ".venv/Scripts/activate" ]; then          # Windows Git Bash / WSL
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then             # Linux / macOS
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  err "Cannot locate venv activation script."
fi
ok "Virtual environment activated"

# ── 3. Install / upgrade dependencies ────────────────────────────────────────
step "Installing dependencies from requirements.txt…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "All dependencies installed"

# ── 4. Dataset check ─────────────────────────────────────────────────────────
step "Verifying dataset…"
CSV_PATH="data/PS_20174392719_1491204439457_log.csv"
if [ ! -f "$CSV_PATH" ]; then
  err "Dataset not found at $CSV_PATH. Download from Kaggle: paysim1"
fi
LINES=$(wc -l < "$CSV_PATH")
ok "Dataset found — $LINES lines"

# ── 5. Unit tests ─────────────────────────────────────────────────────────────
step "Running unit tests (pytest)…"
if pytest tests/ -v --tb=short 2>&1; then
  ok "All tests passed"
else
  echo -e "${YELLOW}  ⚠  Some tests failed — check output above.${RESET}"
  echo -e "${YELLOW}  Continuing to launch the application…${RESET}"
fi

# ── 6. Launch Streamlit ───────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  🚀 Launching SentinelPay…  http://localhost:8501${RESET}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "  Press ${BOLD}Ctrl+C${RESET} to stop the server."
echo ""

streamlit run app.py \
  --server.headless true \
  --server.port 8501 \
  --browser.gatherUsageStats false
