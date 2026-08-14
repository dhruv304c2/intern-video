#!/bin/bash
# start/stop helper for the other shell scripts - sourced, not run directly.
# Backgrounds a command under a pid file in .pids/, keyed by name, so a
# second `start` refuses to double-launch and `stop` can find it later.
PID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.pids"

start_daemon() {
    local name="$1"
    shift
    mkdir -p "$PID_DIR"
    local pidfile="$PID_DIR/$name.pid"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "$name already running (pid $(cat "$pidfile"))" >&2
        exit 1
    fi
    nohup "$@" > "$PID_DIR/$name.log" 2>&1 &
    echo $! > "$pidfile"
    echo "$name started (pid $!) - logs: $PID_DIR/$name.log"
}

stop_daemon() {
    local name="$1"
    local pidfile="$PID_DIR/$name.pid"
    if [ ! -f "$pidfile" ] || ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "$name is not running" >&2
        rm -f "$pidfile"
        return 0
    fi
    kill "$(cat "$pidfile")"
    rm -f "$pidfile"
    echo "$name stopped"
}
