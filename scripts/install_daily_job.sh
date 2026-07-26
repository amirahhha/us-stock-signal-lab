#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$PROJECT_DIR/scripts/com.amirah.us-stock-signal-lab.plist.template"
TARGET="$HOME/Library/LaunchAgents/com.amirah.us-stock-signal-lab.plist"
DOMAIN="gui/$(id -u)"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"
sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$TEMPLATE" > "$TARGET"
chmod +x "$PROJECT_DIR/scripts/run_daily_refresh.sh"

launchctl bootout "$DOMAIN" "$TARGET" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$TARGET"
launchctl enable "$DOMAIN/com.amirah.us-stock-signal-lab"

echo "Installed weekday refresh at 6:00 PM local time."
echo "LaunchAgent: $TARGET"
echo "Log: $PROJECT_DIR/logs/daily_refresh.log"
