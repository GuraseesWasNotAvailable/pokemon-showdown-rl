#!/usr/bin/env bash
# Sets up the training box. Runs ON the box (Ubuntu 24.04 with the NVIDIA driver image), as the
# ubuntu user, after the repo has been copied to ~/ps-ppo (cloud/vm.sh deploy does that).
# Safe to run again: every step checks whether it is already done.
#
#   bash ~/ps-ppo/cloud/bootstrap.sh
#
# What it leaves behind:
#   ~/pokemon-showdown   the Showdown server at the commit the laptop uses, built, with the
#                        rlspawn plugin and bound to 127.0.0.1
#   ~/venv               Python with the same torch, ray, poke-env and numpy as the laptop
#   ~/runs/main          the run directory (checkpoints, logs, evaluation results)
set -euo pipefail

REPO="$HOME/ps-ppo"
PS="$HOME/pokemon-showdown"
VENV="$HOME/venv"
PS_COMMIT=3661ce40bf9001d185ce078b8920e12304204609
NODE_MAJOR=24

log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }

if [ ! -f "$REPO/train.py" ]; then echo "copy the repo to $REPO first (cloud/vm.sh deploy)"; exit 1; fi

log "waiting for first-boot setup to finish"
sudo cloud-init status --wait >/dev/null 2>&1 || true

log "system packages"
export DEBIAN_FRONTEND=noninteractive
sudo -E apt-get update -y -q
sudo -E apt-get install -y -q git curl ca-certificates build-essential python3-venv python3-dev \
    tmux htop jq unzip pkg-config libssl-dev

if ! command -v node >/dev/null 2>&1 || [ "$(node -v | cut -d. -f1)" != "v$NODE_MAJOR" ]; then
  log "node $NODE_MAJOR"
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | sudo -E bash - >/dev/null
  sudo -E apt-get install -y -q nodejs
fi
log "node $(node -v), npm $(npm -v)"

if [ ! -d "$PS/.git" ]; then
  log "cloning the Showdown server"
  git clone -q https://github.com/smogon/pokemon-showdown.git "$PS"
fi
cd "$PS"
if [ "$(git rev-parse HEAD)" != "$PS_COMMIT" ]; then
  git fetch -q origin
  git checkout -q "$PS_COMMIT"
fi
# Only when it differs: an unconditional copy makes the plugin newer than its built file every
# time, and the build below then rewrites dist/ under whatever servers are running.
cmp -s "$REPO/pokemon-showdown/server/chat-plugins/rlspawn.ts" server/chat-plugins/rlspawn.ts \
  || cp "$REPO/pokemon-showdown/server/chat-plugins/rlspawn.ts" server/chat-plugins/rlspawn.ts
if [ ! -f config/config.js ]; then
  cp config/config-example.js config/config.js
  printf "\n// training box: listen on loopback only\nexports.bindaddress = '127.0.0.1';\n" >> config/config.js
fi
if [ ! -d node_modules ]; then
  log "npm install (Showdown)"
  npm ci --no-audit --no-fund >/dev/null 2>&1 || npm install --no-audit --no-fund
fi
if [ ! -f dist/server/chat-plugins/rlspawn.js ] || [ server/chat-plugins/rlspawn.ts -nt dist/server/chat-plugins/rlspawn.js ]; then
  # A build rewrites dist/ under the servers that run from it (start_servers.py starts them with
  # --skip-build for that reason). Not while any training or evaluation server is listening.
  LISTENING=$(ss -ltnH 2>/dev/null | awk '{print $4}' | grep -E ':(80[0-9][0-9]|9000)$' || true)
  if [ -n "$LISTENING" ]; then
    echo "the Showdown server needs a build, but servers are listening ($(echo $LISTENING)). Stop the run first (sudo systemctl stop psppo-train psppo-eval, or the run by hand), then run this again."
    exit 1
  fi
  log "building the Showdown server"
  node build
fi
test -f dist/server/chat-plugins/rlspawn.js || { echo "the rlspawn plugin did not build"; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
  log "python environment"
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip wheel
# The versions the laptop trained and tested with (2026-10-05).
"$VENV/bin/pip" install -q "torch==2.12.0" "numpy==2.4.6" "poke-env==0.15.0" "ray==2.59.0" psutil py-spy pytest

log "PSBot engine data"
(cd "$REPO" && node heuristic/setup.js | tail -3)

mkdir -p "$HOME/runs/main"

# Foul Play, the outside benchmark opponent and the search behind the bridge. Its own
# virtualenv under ~/foul-play; a second run is a no-op.
log "Foul Play"
bash "$REPO/bench/install_foulplay.sh" > "$HOME/foulplay_install.log" 2>&1 || { tail -5 "$HOME/foulplay_install.log"; echo "Foul Play did not install (see ~/foulplay_install.log)"; exit 1; }
tail -1 "$HOME/foulplay_install.log"

log "checks"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$VENV/bin/python" - <<'PY'
import torch, ray, poke_env, numpy, importlib.metadata as m
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("ray", ray.__version__, "poke-env", m.version("poke-env"), "numpy", numpy.__version__)
import triton
print("triton", triton.__version__)
assert torch.cuda.is_available(), "no GPU visible to torch"
PY
echo "cores: $(nproc)   ram: $(free -g | awk '/Mem:/{print $2}') GB   disk free: $(df -h --output=avail "$HOME" | tail -1)"
log "done"
