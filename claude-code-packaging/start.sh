#!/bin/bash
# start.sh — run the LiteLLM proxy in the foreground on 127.0.0.1:4000.
# Leave this running in one terminal; use `alice-claude` from another.
#
# Provider selection (which upstream LLM the proxy calls):
#   ./start.sh                  # uses PROVIDER from ~/.alice-litellm/.env
#   ./start.sh --bedrock        # force AWS Bedrock Claude Opus 4.7
#   ./start.sh --anthropic      # force Anthropic API direct

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INSTALL_DIR="${ALICE_LITELLM_HOME:-$HOME/.alice-litellm}"
VENV_DIR="$INSTALL_DIR/venv"
ENV_FILE="$INSTALL_DIR/.env"

# --- Parse flags ----------------------------------------------------------

PROVIDER_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --bedrock)   PROVIDER_OVERRIDE="bedrock"; shift ;;
        --anthropic) PROVIDER_OVERRIDE="anthropic"; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *) echo "[alice-litellm] unknown flag: $1" >&2; exit 2 ;;
    esac
done

# --- Sanity checks --------------------------------------------------------

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

# --- Load env vars --------------------------------------------------------

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Apply runtime override
if [ -n "$PROVIDER_OVERRIDE" ]; then
    PROVIDER="$PROVIDER_OVERRIDE"
    echo "[alice-litellm] provider override (this run only): $PROVIDER"
fi

if [ -z "${PROVIDER:-}" ]; then
    echo "[alice-litellm] PROVIDER not set in $ENV_FILE and no --bedrock/--anthropic flag given." >&2
    echo "[alice-litellm] Re-run ./install.sh or pass --bedrock / --anthropic." >&2
    exit 1
fi

# --- Validate provider-specific secrets ----------------------------------

if [ -z "${WONDERFENCE_API_KEY:-}" ] || [ -z "${WONDERFENCE_APP_ID:-}" ]; then
    echo "[alice-litellm] WONDERFENCE_API_KEY or WONDERFENCE_APP_ID missing in $ENV_FILE" >&2
    exit 1
fi

case "$PROVIDER" in
    bedrock)
        CONFIG_FILE="$SCRIPT_DIR/litellm-config-bedrock.yaml"
        : "${AWS_REGION:=us-west-2}"
        export AWS_REGION
        if ! "$VENV_DIR/bin/python" -c "import boto3; boto3.client('sts').get_caller_identity()" 2>/dev/null; then
            echo "[alice-litellm] AWS credentials not available or expired." >&2
            echo "[alice-litellm] Run: login_dev" >&2
            echo "    (pritunl VPN + AWS SSO refresh — alias defined in af-config)" >&2
            exit 1
        fi
        ;;
    anthropic)
        CONFIG_FILE="$SCRIPT_DIR/litellm-config-anthropic.yaml"
        if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
            echo "[alice-litellm] ANTHROPIC_API_KEY missing in $ENV_FILE" >&2
            echo "[alice-litellm] Add it or re-run ./install.sh." >&2
            exit 1
        fi
        ;;
    *)
        echo "[alice-litellm] Unknown PROVIDER='$PROVIDER' (expected: bedrock | anthropic)" >&2
        exit 1
        ;;
esac

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[alice-litellm] Config not found: $CONFIG_FILE" >&2
    exit 1
fi

# --- Launch ---------------------------------------------------------------

echo "[alice-litellm] Starting LiteLLM proxy on 127.0.0.1:4000"
echo "[alice-litellm] provider: $PROVIDER"
echo "[alice-litellm] config:   $CONFIG_FILE"
echo "[alice-litellm] WONDERFENCE_APP_ID: $WONDERFENCE_APP_ID"
[ "$PROVIDER" = "bedrock" ] && echo "[alice-litellm] AWS_REGION: $AWS_REGION"
echo "[alice-litellm] Press Ctrl+C to stop."
echo

exec "$VENV_DIR/bin/litellm" \
    --config "$CONFIG_FILE" \
    --host 127.0.0.1 \
    --port 4000
