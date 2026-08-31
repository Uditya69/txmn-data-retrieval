#!/usr/bin/env bash
# Thin headless runner for the eval CLIs (slm_intent_eval, intent_eval,
# collection_routing_eval, retrieval_eval) - not a new orchestration framework, just
# enough to survive an SSH disconnect during a multi-hour run: background the process,
# redirect output to a log file, track it by PID.
#
# Usage:
#   evals/run_headless.sh start <name> -- <command...>
#   evals/run_headless.sh status <name>
#   evals/run_headless.sh stop <name>
#
# Example:
#   evals/run_headless.sh start slm-intent -- \
#     uv run python -m retrieval_api.slm_intent_eval \
#       --dataset evals/slm_intent_cases.json \
#       --output .eval-results/slm-intent.jsonl --resume
#   evals/run_headless.sh status slm-intent
#   evals/run_headless.sh stop slm-intent

set -euo pipefail

DIR=".eval-results/headless"

usage() {
    echo "usage: $0 {start|status|stop} <name> [-- <command...>]" >&2
    exit 1
}

[ $# -ge 2 ] || usage
action="$1"
name="$2"
shift 2

pid_file="$DIR/$name.pid"
log_file="$DIR/$name.log"

case "$action" in
    start)
        [ "${1:-}" = "--" ] || usage
        shift
        [ $# -ge 1 ] || usage
        mkdir -p "$DIR"
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo "already running: $name (pid $(cat "$pid_file"))" >&2
            exit 1
        fi
        nohup "$@" > "$log_file" 2>&1 < /dev/null &
        echo $! > "$pid_file"
        disown
        echo "started: $name (pid $(cat "$pid_file")), log: $log_file"
        ;;
    status)
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo "running: $name (pid $(cat "$pid_file"))"
        else
            echo "not running: $name"
            exit 1
        fi
        ;;
    stop)
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            kill "$(cat "$pid_file")"
            echo "stopped: $name (pid $(cat "$pid_file"))"
        else
            echo "not running: $name"
            exit 1
        fi
        ;;
    *)
        usage
        ;;
esac
