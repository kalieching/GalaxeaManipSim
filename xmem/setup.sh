#!/usr/bin/env bash
# Set up xmem: G0.5 VLA (GalaxeaVLA) + GalaxeaManipSim, side by side in one folder.
#
#   xmem/                     <- the folder that holds both repos
#   ├── GalaxeaManipSim/      <- this repo (branch xmem/cube-bowl)
#   └── GalaxeaVLA/           <- cloned here by this script, pinned to VLA_COMMIT
#
# Usage: bash GalaxeaManipSim/xmem/setup.sh
# Safe to re-run: each step is skipped when it is already done.
# Everything goes inside the xmem folder, except uv's own cache and Python downloads (~/.cache/uv, ~/.local/share/uv).
set -euo pipefail

VLA_REPO=https://github.com/OpenGalaxea/GalaxeaVLA.git
VLA_COMMIT=89f2322
PYTHON=3.10.16  # GalaxeaVLA requires >=3.10.16,<3.11, and the sim uses the same interpreter
TORCH=2.7.1     # same build as GalaxeaVLA's lock, so uv reuses one cached copy

SIM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$SIM")"
VLA="$ROOT/GalaxeaVLA"
CKPT="$VLA/checkpoints"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

step "Checking tools"
command -v git >/dev/null || die "git is not installed"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: an NVIDIA GPU and driver are required"
command -v uv >/dev/null || die "uv is not installed. Install it with:
  curl -LsSf https://astral.sh/uv/install.sh | sh
then open a new shell and re-run this script."
if ! command -v cmake >/dev/null; then
  echo "cmake not found, installing it as a uv tool (into ~/.local/bin)"
  uv tool install cmake
fi
echo "ok: git, nvidia-smi, uv $(uv --version | cut -d' ' -f2), cmake"

step "GalaxeaVLA at $VLA_COMMIT"
if [ ! -d "$VLA/.git" ]; then
  git clone "$VLA_REPO" "$VLA"
  git -C "$VLA" checkout -q "$VLA_COMMIT"
fi
head=$(git -C "$VLA" rev-parse --short=7 HEAD)
if [ "$head" != "$VLA_COMMIT" ]; then
  echo "warning: $VLA is at $head, not $VLA_COMMIT (left as is, run: git -C $VLA checkout $VLA_COMMIT)"
else
  echo "ok: $head"
fi

step "VLA environment ($VLA/.venv, ~11 GB)"
(cd "$VLA" && CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --python "$PYTHON" --index-strategy unsafe-best-match)
# uv sync rewrites index URLs in uv.lock, a change that is not ours to commit
echo "ok (note: GalaxeaVLA/uv.lock shows as modified after uv sync, don't commit it)"

step "Simulator environment ($SIM/.venv, ~6 GB)"
[ -x "$SIM/.venv/bin/python" ] || uv venv --python "$PYTHON" "$SIM/.venv"
# numpy<2: mplib 0.2.1 segfaults on numpy 2. setuptools<81: sapien imports pkg_resources.
uv pip install --python "$SIM/.venv/bin/python" --torch-backend cu128 \
  -e "$SIM" "numpy<2" "setuptools<81" websockets msgpack imageio-ffmpeg "torch==$TORCH"
echo "ok"

step "Simulator object meshes (~600 MB; parts already there are skipped)"
MODELS="$SIM/galaxea_sim/assets/robotwin_models"
if [ -L "$MODELS" ] && [ ! -e "$MODELS" ]; then
  die "$MODELS is a symlink to a path that does not exist ($(readlink "$MODELS")). Remove the link and re-run."
fi
"$SIM/.venv/bin/python" -m galaxea_sim.scripts.fetch_assets

step "G0.5 weights ($CKPT, ~12 GB)"
if [ -f "$CKPT/g05-base/checkpoints/model_state_dict.pt" ] && [ -f "$CKPT/action_tokenizer.pt" ] \
   && [ -d "$CKPT/qwen3_5_2b_base_processor" ]; then
  echo "ok: already present"
else
  HF="$VLA/.venv/bin/hf"
  if ! "$HF" auth whoami >/dev/null 2>&1; then
    die "not logged in to Hugging Face. First accept the license at https://huggingface.co/OpenGalaxea/G05
(gated, approved instantly), then run:
  $HF auth login    # paste a read token from https://huggingface.co/settings/tokens
and re-run this script."
  fi
  "$HF" download OpenGalaxea/G05 --local-dir "$CKPT" \
    --include "g05-base/*" --include "action_tokenizer.pt" --include "qwen3_5_2b_base_processor/*"
fi

step "Done"
echo "Start the sim:   bash $SIM/xmem/up.sh --robot r1lite"
echo "Or for a robot:  bash $SIM/xmem/up.sh --robot r1pro --real"
