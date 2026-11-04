#!/usr/bin/env bash
# Builds Assignment03_SuarezNavarro_Report.pdf from report/report.md
# (pandoc -> self-contained HTML -> headless Chromium "print to PDF").
# Set CHROME to your Chrome/Chromium binary if it is not found automatically.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-$(ls ~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-mac/headless_shell 2>/dev/null | head -1)}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
tmp="$(mktemp -d)"
(cd report && pandoc report.md -s --css style.css --self-contained --metadata lang=en -o "$tmp/report.html")
"$CHROME" --headless --no-pdf-header-footer --print-to-pdf="$PWD/Assignment03_SuarezNavarro_Report.pdf" "file://$tmp/report.html" 2>/dev/null
rm -rf "$tmp"
echo "Assignment03_SuarezNavarro_Report.pdf written"
