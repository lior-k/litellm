#!/bin/bash
# install.sh — one-time setup for the alice-claude / Claude Code + WonderFence
# guardrail toolchain on macOS.
#
# Creates an isolated uv venv at ~/.alice-litellm/venv, installs deps, asks
# which upstream LLM provider to use (Bedrock or Anthropic direct), prompts
# for the matching credentials, and symlinks `alice-claude` onto PATH.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${ALICE_LITELLM_HOME:-$HOME/.alice-litellm}"
VENV_DIR="$INSTALL_DIR/venv"
ENV_FILE="$INSTALL_DIR/.env"

echo "[alice-litellm] install.sh starting"
echo "[alice-litellm] install dir: $INSTALL_DIR"
echo

# --- Helpers --------------------------------------------------------------

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

set_kv() {
    local var="$1"
    local value="$2"
    if grep -q "^$var=" "$ENV_FILE" 2>/dev/null; then
        # macOS sed in-place
        sed -i '' -E "s|^$var=.*|$var=$value|" "$ENV_FILE"
    else
        echo "$var=$value" >> "$ENV_FILE"
    fi
}

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

# --- Step 3: provider choice ---------------------------------------------

touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

CURRENT_PROVIDER=$(grep -E "^PROVIDER=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- || true)
if [ -n "$CURRENT_PROVIDER" ]; then
    echo "[alice-litellm] PROVIDER already set to '$CURRENT_PROVIDER' (re-using)"
    PROVIDER="$CURRENT_PROVIDER"
else
    echo
    echo "Which upstream LLM provider should the proxy call?"
    echo "  1) bedrock   — AWS Bedrock Claude Opus 4.7 (us-west-2). Needs AWS SSO via login_dev."
    echo "  2) anthropic — Anthropic API direct (latest Opus, Sonnet, Haiku). Needs a real Anthropic API key."
    echo
    read -r -p "Choice [1/2] (no default): " choice
    case "$choice" in
        1|bedrock)   PROVIDER="bedrock" ;;
        2|anthropic) PROVIDER="anthropic" ;;
        *) echo "[alice-litellm] Invalid choice; aborting." >&2; exit 1 ;;
    esac
    set_kv PROVIDER "$PROVIDER"
fi

echo "[alice-litellm] provider: $PROVIDER"

# --- Step 4: secrets ------------------------------------------------------

prompt_secret WONDERFENCE_API_KEY "WonderFence API key"
prompt_secret WONDERFENCE_APP_ID  "WonderFence App ID (UUID)"

case "$PROVIDER" in
    bedrock)
        if ! grep -q "^AWS_REGION=" "$ENV_FILE" 2>/dev/null; then
            echo "AWS_REGION=us-west-2" >> "$ENV_FILE"
        fi
        ;;
    anthropic)
        prompt_secret ANTHROPIC_API_KEY "Anthropic API key (sk-ant-...)"
        ;;
esac

# --- Step 5: provider-specific preflight ---------------------------------

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

case "$PROVIDER" in
    bedrock)
        if ! "$VENV_DIR/bin/python" -c "import boto3; boto3.client('sts').get_caller_identity()" 2>/dev/null; then
            echo
            echo "[alice-litellm] AWS credentials not available or expired."
            echo "[alice-litellm] Run this before ./start.sh:"
            echo "    login_dev"
            echo "    (pritunl VPN + AWS SSO refresh)"
            echo
        fi
        ;;
    anthropic)
        echo "[alice-litellm] (provider=anthropic — skipping AWS preflight)"
        ;;
esac

# --- Step 6: symlink alice-claude ----------------------------------------

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

[alice-litellm] install complete. provider=$PROVIDER

Next steps:
EOF
if [ "$PROVIDER" = "bedrock" ]; then
    cat <<EOF
  1. (if AWS expired) login_dev
  2. Terminal A:  cd $SCRIPT_DIR && ./start.sh
  3. Terminal B:  alice-claude

To switch to Anthropic at any time:  ./start.sh --anthropic
EOF
else
    cat <<EOF
  1. Terminal A:  cd $SCRIPT_DIR && ./start.sh
  2. Terminal B:  alice-claude

To switch to Bedrock at any time:  ./start.sh --bedrock  (needs AWS creds)
EOF
fi
echo
