#!/bin/zsh
set -euo pipefail

TARGET="$HOME/Library/LaunchAgents/com.amirah.us-stock-signal-lab.plist"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN" "$TARGET" 2>/dev/null || true
if [[ -f "$TARGET" ]]; then
  mv "$TARGET" "$HOME/.Trash/com.amirah.us-stock-signal-lab.plist"
fi

echo "Daily refresh job removed. Existing database and logs were not deleted."
