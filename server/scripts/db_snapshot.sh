#!/bin/bash
# Consistent snapshots of the live tinker SQLite DB back to durable storage.
#
# WHY THIS EXISTS: the tinker DB corrupts when run directly on /work1 (wekafs).
# SQLite's WAL mode needs POSIX fcntl locking + mmap shared memory that network
# filesystems don't provide reliably -- job 347512 (2026-07-14) died with
# "database disk image is malformed". So run.sh keeps the *live* DB on the
# node-local disk (ext4/xfs, real locking) and this script periodically copies a
# consistent snapshot back to durable /work1 storage.
#
# The copy uses `sqlite3 .backup` (the online backup API), which is safe to run
# WHILE the server writes -- a plain `cp` of a live WAL database can capture a
# torn state. The snapshot is written to a temp name on the durable filesystem
# and atomically renamed, so a reader on /work1 never sees a half-written file.
#
# Periodic snapshots matter because the node-local DB vanishes with the node: a
# scancel / OOM / walltime kill sends SIGKILL, and a copy-back-on-exit trap does
# NOT run for SIGKILL. Snapshotting every few minutes bounds the loss to one
# interval instead of the whole run.
#
# Usage:
#   db_snapshot.sh once <live_db> <durable_db>              # single snapshot
#   db_snapshot.sh loop <live_db> <durable_db> [interval]   # every <interval>s
#                                                           # (default 300) until
#                                                           # SIGTERM/SIGINT, then
#                                                           # one final snapshot
set -uo pipefail

snapshot() {
    local live="$1" durable="$2"
    # Nothing to snapshot until the server has created the DB.
    [ -f "$live" ] || return 0
    local tmp="${durable}.snap.$$"
    mkdir -p "$(dirname "$durable")"
    # .backup reads a consistent view (main DB + WAL) even under concurrent
    # writes. On success, atomically swap it in; on failure, leave the previous
    # good snapshot untouched.
    if sqlite3 "$live" ".backup '$tmp'" 2>/dev/null; then
        mv -f "$tmp" "$durable"
        return 0
    fi
    rm -f "$tmp"
    return 1
}

mode="${1:-}"
LIVE="${2:-}"
DURABLE="${3:-}"

if [ -z "$mode" ] || [ -z "$LIVE" ] || [ -z "$DURABLE" ]; then
    echo "usage: $0 once|loop <live_db> <durable_db> [interval_sec]" >&2
    exit 2
fi

case "$mode" in
    once)
        if snapshot "$LIVE" "$DURABLE"; then
            echo "snapshot: $LIVE -> $DURABLE ok"
        else
            echo "snapshot: $LIVE -> $DURABLE FAILED (live db missing or backup error)" >&2
            exit 1
        fi
        ;;
    loop)
        INTERVAL="${4:-300}"
        # On termination, take one last snapshot so a graceful stop loses nothing.
        _stop=0
        trap '_stop=1' TERM INT
        echo "db_snapshot: looping every ${INTERVAL}s ($LIVE -> $DURABLE)"
        while [ "$_stop" -eq 0 ]; do
            snapshot "$LIVE" "$DURABLE" || echo "db_snapshot: snapshot skipped/failed" >&2
            # Sleep in 1s slices so a signal interrupts promptly.
            for _ in $(seq 1 "$INTERVAL"); do
                [ "$_stop" -eq 1 ] && break
                sleep 1
            done
        done
        snapshot "$LIVE" "$DURABLE" && echo "db_snapshot: final snapshot ok"
        ;;
    *)
        echo "usage: $0 once|loop <live_db> <durable_db> [interval_sec]" >&2
        exit 2
        ;;
esac
