#!/bin/bash
# start.sh — run the LiteLLM proxy in the foreground on 127.0.0.1:4000.
# Leave this running in one terminal; use `alice-claude` from another.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INSTALL_DIR="${ALICE_LITELLM_HOME:-$HOME/.alice-litellm}"
VENV_DIR="$INSTALL_DIR/venv"
ENV_FILE="$INSTALL_DIR/.env"

# --- Sanity checks ---------------------------------------------------------

if [ ! -d "$VENV_DIR" ]; then
    echo "[alice-litellm] venv not found at $VENV_DIR" >&2
    echo "[alice-litellm] Run ./install.sh first." >&2
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "[alice-litellm] .env not found at $ENV_FILE" >&2
    echo "[alice-litellm] Run ./install.sh first." >&2
    exit 1
fi

# --- Load env vars ---------------------------------------------------------

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${AWS_REGION:=us-west-2}"
export AWS_REGION

if [ -z "${WONDERFENCE_API_KEY:-}" ] || [ -z "${WONDERFENCE_APP_ID:-}" ]; then
    echo "[alice-litellm] WONDERFENCE_API_KEY or WONDERFENCE_APP_ID missing in $ENV_FILE" >&2
    exit 1
fi

# --- AWS creds check ------------------------------------------------------

if ! "$VENV_DIR/bin/python" -c "import boto3, sys; \
sts=boto3.client('sts'); sts.get_caller_identity()" 2>/dev/null; then
    echo "[alice-litellm] AWS credentials not available or expired." >&2
    echo "[alice-litellm] Run: login_dev" >&2
    echo "    (pritunl VPN + AWS SSO refresh — alias defined in af-config)" >&2
    exit 1
fi

# --- Launch ----------------------------------------------------------------

echo "[alice-litellm] Starting LiteLLM proxy on 127.0.0.1:4000"
echo "[alice-litellm] WONDERFENCE_APP_ID: $WONDERFENCE_APP_ID"
echo "[alice-litellm] AWS_REGION: $AWS_REGION"
echo "[alice-litellm] Press Ctrl+C to stop."
echo

exec "$VENV_DIR/bin/litellm" \
    --config "$SCRIPT_DIR/litellm-config.yaml" \
    --host 127.0.0.1 \
    --port 4000
