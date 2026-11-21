#!/usr/bin/env bash
# Start the G0.5 policy server, and (unless --real) the simulator with its web page.
#
# Usage: bash GalaxeaManipSim/xmem/up.sh [--robot r1lite|r1pro] [--real | --server-only] [--gpu N]
#                                        [--port 8765] [--web-port 8000] [--max-steps 1200]
#   --robot      embodiment for both the server and the sim (default r1lite)
#   --real       start only the policy server and print how to connect a real robot to it
#   --server-only  start only the policy server (e.g. for xmem/collect_kitting.py)
#   --gpu        GPU index for the server (default: the GPU with the most free memory)
#   --port       policy server port (default 8765)
#   --web-port   sim web page port (default 8000)
#   --max-steps  sim steps per prompt, at 15 Hz (default 1200)
#
# Both servers bind to 127.0.0.1 only, since they have no authentication. Ctrl+C stops everything this script started.
# Run setup.sh first. Logs go to xmem/logs/.
set -euo pipefail

ROBOT=r1lite REAL=0 SERVER_ONLY=0 GPU="" PORT=8765 WEB_PORT=8000 MAX_STEPS=1200
MIN_FREE_MIB=16000  # G0.5 server uses ~16.5 GB

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --robot) ROBOT="${2:?}"; shift 2 ;;
    --real) REAL=1; shift ;;
    --server-only) SERVER_ONLY=1; shift ;;
    --gpu) GPU="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --web-port) WEB_PORT="${2:?}"; shift 2 ;;
    --max-steps) MAX_STEPS="${2:?}"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done
case "$ROBOT" in r1lite|r1pro) ;; *) die "--robot must be r1lite or r1pro" ;; esac

SIM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$SIM")"
VLA="$ROOT/GalaxeaVLA"
CKPT="$VLA/checkpoints/g05-base/checkpoints/model_state_dict.pt"
LOGS="$ROOT/logs"

[ -x "$VLA/.venv/bin/python" ] || die "VLA environment missing; run: bash $SIM/xmem/setup.sh"
[ -f "$CKPT" ] || die "weights missing ($CKPT); run: bash $SIM/xmem/setup.sh"
[ "$REAL" = 1 ] || [ -x "$SIM/.venv/bin/python" ] || die "sim environment missing; run: bash $SIM/xmem/setup.sh"

port_busy() { [ -n "$(ss -Hltn "sport = :$1")" ]; }
port_busy "$PORT" && die "port $PORT is already in use (possibly by another user). Pick another with --port."
[ "$REAL" = 1 ] || [ "$SERVER_ONLY" = 1 ] || ! port_busy "$WEB_PORT" || \
  die "port $WEB_PORT is already in use. Pick another with --web-port."

# Pick the GPU with the most free memory. Refuse one without room, so we don't crowd anyone else's jobs.
if [ -z "$GPU" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
fi
free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits) || die "no GPU with index $GPU"
[ "$free" -ge "$MIN_FREE_MIB" ] || die "GPU $GPU has only ${free} MiB free (need ~${MIN_FREE_MIB}). Current use:
$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv)"

mkdir -p "$LOGS"
PIDS=()
cleanup() {
  trap - EXIT INT TERM
  # each child runs in its own process group (setsid), so this stops it and anything it spawned
  for pid in "${PIDS[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "stopped"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# wait_for <pid> <log> <pattern> <seconds>: wait until the log shows the pattern, or fail if the process exits
wait_for() {
  local pid=$1 log=$2 pattern=$3 secs=$4
  for ((i = 0; i < secs; i++)); do
    grep -q "$pattern" "$log" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || { echo "--- last lines of $log:" >&2; tail -20 "$log" >&2; die "process exited before it was ready"; }
    sleep 1
  done
  die "not ready after ${secs}s; see $log"
}

SERVER_LOG="$LOGS/server-$ROBOT.log"
echo "Starting G0.5 policy server ($ROBOT) on GPU $GPU, 127.0.0.1:$PORT (1-2 min)... log: $SERVER_LOG"
(cd "$VLA" && CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=src:scripts exec setsid .venv/bin/python \
  "$SIM/xmem/serve_policy_5090.py" --ckpt_path "$CKPT" --host 127.0.0.1 --port "$PORT" \
  "eval_embodiment=galaxea_$ROBOT" model.model_arch.attn_implementation=sdpa \
  model.model_arch.discrete_action=true model.model_arch.continuous_action=false) >"$SERVER_LOG" 2>&1 &
PIDS+=($!)
wait_for "${PIDS[0]}" "$SERVER_LOG" "Policy server listening" 600
echo "Policy server ready."

if [ "$REAL" = 1 ]; then
  cat <<EOF

Connect the $ROBOT (ROS2 Humble) to this server. On the robot's computer:
  1. Tunnel the robot's port 8080 (its client's default) to this server, and leave it open:
       ssh -N -L 8080:127.0.0.1:$PORT $(whoami)@$(hostname -f)
  2. Start the client. The default config.toml (127.0.0.1:8080) needs no changes:
       source /opt/ros/humble/setup.bash
       cd GalaxeaVLA/experiments/$ROBOT && python run.py --config config.toml
     See GalaxeaVLA/experiments/$ROBOT/README.md for topics, instructions, and recording.

Ctrl+C here stops the server.
EOF
elif [ "$SERVER_ONLY" = 1 ]; then
  echo "Policy server at ws://127.0.0.1:$PORT. Ctrl+C here stops it."
else
  CLIENT_LOG="$LOGS/sim-$ROBOT.log"
  echo "Starting simulator and web page... log: $CLIENT_LOG"
  (cd "$SIM" && exec setsid .venv/bin/python xmem/vla_client.py --robot "$ROBOT" \
    --server "ws://127.0.0.1:$PORT" --port "$WEB_PORT" --max-steps "$MAX_STEPS") >"$CLIENT_LOG" 2>&1 &
  PIDS+=($!)
  wait_for "${PIDS[1]}" "$CLIENT_LOG" "Open http" 300
  cat <<EOF

Ready: open http://localhost:$WEB_PORT
  If this machine is remote, forward port $WEB_PORT first (VS Code: Ports panel -> Forward a Port -> $WEB_PORT;
  or on your laptop: ssh -L $WEB_PORT:127.0.0.1:$WEB_PORT $(whoami)@$(hostname -f)).
Ctrl+C here stops the server and the simulator.
EOF
fi

# Stay in the foreground until Ctrl+C or until either process exits.
while :; do
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || die "a process exited unexpectedly; see $LOGS/"
  done
  sleep 2
done
