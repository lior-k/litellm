#!/bin/bash
# install.sh — one-time setup for the alice-claude / Claude Code + WonderFence
# guardrail toolchain on macOS.
#
# Creates an isolated uv venv at ~/.alice-litellm/venv, installs deps,
# prompts for WonderFence credentials, verifies AWS creds, and symlinks
# `alice-claude` onto PATH.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${ALICE_LITELLM_HOME:-$HOME/.alice-litellm}"
VENV_DIR="$INSTALL_DIR/venv"
ENV_FILE="$INSTALL_DIR/.env"

echo "[alice-litellm] install.sh starting"
echo "[alice-litellm] install dir: $INSTALL_DIR"
echo

# --- Step 1: uv -----------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "[alice-litellm] uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[alice-litellm] uv: $(uv --version)"

# --- Step 2: venv + deps --------------------------------------------------

mkdir -p "$INSTALL_DIR"
if [ ! -d "$VENV_DIR" ]; then
    echo "[alice-litellm] Creating venv at $VENV_DIR"
    uv venv "$VENV_DIR"
fi

echo "[alice-litellm] Installing dependencies..."
VIRTUAL_ENV="$VENV_DIR" uv pip install -r "$SCRIPT_DIR/requirements.txt"

# --- Step 3: secrets ------------------------------------------------------

touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

prompt_secret() {
    local var="$1"
    local prompt="$2"
    local existing
    existing=$(grep -E "^$var=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- || true)
    if [ -n "$existing" ]; then
        echo "[alice-litellm] $var already set (re-using existing value)"
        return
    fi
    read -r -p "$prompt: " value
    if [ -z "$value" ]; then
        echo "[alice-litellm] $var is required; aborting." >&2
        exit 1
    fi
    echo "$var=$value" >> "$ENV_FILE"
}

prompt_secret WONDERFENCE_API_KEY "WonderFence API key"
prompt_secret WONDERFENCE_APP_ID  "WonderFence App ID (UUID)"

if ! grep -q "^AWS_REGION=" "$ENV_FILE" 2>/dev/null; then
    echo "AWS_REGION=us-west-2" >> "$ENV_FILE"
fi

# --- Step 4: AWS creds check ---------------------------------------------

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if ! "$VENV_DIR/bin/python" -c "import boto3; boto3.client('sts').get_caller_identity()" 2>/dev/null; then
    echo
    echo "[alice-litellm] AWS credentials not available or expired."
    echo "[alice-litellm] Run this before ./start.sh:"
    echo "    login_dev"
    echo "    (pritunl VPN + AWS SSO refresh)"
    echo
fi

# --- Step 5: symlink alice-claude ----------------------------------------

WRAPPER="$SCRIPT_DIR/bin/alice-claude"
chmod +x "$WRAPPER" "$SCRIPT_DIR/start.sh"

LINK_TARGET=""
if [ -w /usr/local/bin ] 2>/dev/null; then
    LINK_TARGET="/usr/local/bin/alice-claude"
elif [ -w /opt/homebrew/bin ] 2>/dev/null; then
    LINK_TARGET="/opt/homebrew/bin/alice-claude"
else
    mkdir -p "$HOME/.local/bin"
    LINK_TARGET="$HOME/.local/bin/alice-claude"
fi

ln -sfn "$WRAPPER" "$LINK_TARGET"
echo "[alice-litellm] Linked $LINK_TARGET -> $WRAPPER"

case ":$PATH:" in
    *":$(dirname "$LINK_TARGET"):"*) ;;
    *)
        echo "[alice-litellm] WARNING: $(dirname "$LINK_TARGET") is not on PATH."
        echo "[alice-litellm] Add this to ~/.zshrc:"
        echo "    export PATH=\"$(dirname "$LINK_TARGET"):\$PATH\""
        ;;
esac

# --- Done -----------------------------------------------------------------

cat <<EOF

[alice-litellm] install complete.

Next steps:
  1. (if AWS expired) login_dev
  2. Terminal A:  cd $SCRIPT_DIR && ./start.sh
  3. Terminal B:  alice-claude

EOF
