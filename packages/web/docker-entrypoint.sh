#!/bin/sh
set -eu
cat > /usr/share/nginx/html/env-config.js <<EOF
window.__ENV__ = { WS_URL: "${WS_URL:-ws://localhost:8010/ws/search}", AGENT_WS_URL: "${AGENT_WS_URL:-ws://localhost:8010/ws/agent}" };
EOF
