#!/bin/bash
# uninstall.sh — remove the alice-litellm install dir and the alice-claude
# wrapper symlink. Does NOT touch ~/.aws, the cloned fork, or Claude Code.

set -e

INSTALL_DIR="${ALICE_LITELLM_HOME:-$HOME/.alice-litellm}"

echo "[alice-litellm] Removing $INSTALL_DIR (venv + .env + messages)"
read -r -p "Continue? [y/N] " confirm
case "$confirm" in
    y|Y|yes|YES) ;;
    *) echo "[alice-litellm] Aborted."; exit 1 ;;
esac

rm -rf "$INSTALL_DIR"

for path in /usr/local/bin/alice-claude /opt/homebrew/bin/alice-claude "$HOME/.local/bin/alice-claude"; do
    if [ -L "$path" ]; then
        echo "[alice-litellm] Removing symlink $path"
        rm -f "$path"
    fi
done

echo "[alice-litellm] Uninstall complete."
