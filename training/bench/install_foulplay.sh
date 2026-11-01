#!/usr/bin/env bash
#
# Installs Foul Play (https://github.com/pmariglia/foul-play), pinned, as a benchmark opponent.
# Written for Ubuntu 24.04. See bench/FOULPLAY.md.
#
#     bash bench/install_foulplay.sh [DIR]          # DIR defaults to ~/foul-play
#
# Every step is skipped when it is already done, so running this twice is harmless.
#   1. checks git, curl, a C linker and Python 3.11+ with venv, and apt-installs what is missing
#   2. clones Foul Play into DIR at the pinned commit
#   3. installs Rust with rustup when cargo is missing (PATH and dotfiles are left alone)
#   4. creates DIR/.venv (its own virtualenv, nothing goes into the training venv) and installs
#      Foul Play's requirements, building poke-engine from source with Terastallization
#   5. fetches the gen9 random battle set data Foul Play would otherwise download in its first
#      battle, pinned and checksummed
#   6. checks the result: imports, a search that offers a Tera move, `run.py --help`
#
# Environment:
#   PYTHON              interpreter the virtualenv is made from (default: python3)
#   FOULPLAY_COMMIT     Foul Play commit to install (default: the pin below)
#   FOULPLAY_NO_APT=1   never call apt: say what is missing and stop
#   RUST_TOOLCHAIN      toolchain rustup installs when cargo is missing (default: the pin below)
#   RUSTUP_HOME, CARGO_HOME   where Rust lives (rustup's own defaults when unset)
#   CARGO_BUILD_JOBS    cores the engine build may use (default 8: the box may be training)
#
# This script removes nothing. To start over, move DIR away yourself.

set -euo pipefail

# ---- pins ---------------------------------------------------------------------------------
# Foul Play main as of 2026-10-03. Its requirements.txt pins poke-engine 0.0.48.
PINNED_COMMIT="8f541f80f91f960603673796559f796b10ccf0fb"
FOULPLAY_REPO="https://github.com/pmariglia/foul-play.git"
# Rust the engine build was tested with. Only used when cargo is not installed yet.
PINNED_RUST="1.99.0"
# gen9 random battle sets, from github.com/pkmn/randbats at the commit that imported
# smogon/pokemon-showdown@1ce9b34f (2026-10-01). Nothing under data/random-battles/gen9 changed
# between that commit and the server commit this project uses (3661ce4), so the sets match.
SETS_COMMIT="85c017b6570e1ab1b3e047c1e537cdfb064a462a"
SETS_URL="https://raw.githubusercontent.com/pkmn/randbats/${SETS_COMMIT}/data/full/gen9randombattle.json"
SETS_SHA256="6a4b6fc8391d334e252001d8a48ec383159a7ccfdcab23830ab6def00aab793c"
# gen9 random battles need Terastallization. The feature pulls in gen9.
ENGINE_FEATURES="poke-engine/terastallization"
# -------------------------------------------------------------------------------------------

FP_DIR="${1:-$HOME/foul-play}"
COMMIT="${FOULPLAY_COMMIT:-$PINNED_COMMIT}"
PYTHON="${PYTHON:-python3}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-$PINNED_RUST}"

say() { printf '[install_foulplay] %s\n' "$*"; }
die() { printf '[install_foulplay] ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "[install_foulplay] ERROR: stopped at line %s\n" "$LINENO" >&2' ERR

case "$FP_DIR" in
  /*) ;;
  *) FP_DIR="$(pwd)/$FP_DIR" ;;
esac

# ---- 1. prerequisites ---------------------------------------------------------------------
missing=()
command -v git >/dev/null 2>&1 || missing+=(git)
command -v curl >/dev/null 2>&1 || missing+=(curl ca-certificates)
command -v cc >/dev/null 2>&1 || missing+=(build-essential)
command -v sha256sum >/dev/null 2>&1 || missing+=(coreutils)
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  missing+=(python3 python3-venv)
elif ! "$PYTHON" -c "import venv, ensurepip" >/dev/null 2>&1; then
  missing+=(python3-venv)
fi

if [ "${#missing[@]}" -gt 0 ]; then
  if [ "${FOULPLAY_NO_APT:-0}" = "1" ]; then
    die "missing packages: ${missing[*]} (FOULPLAY_NO_APT=1, so nothing was installed)"
  fi
  command -v apt-get >/dev/null 2>&1 || die "missing packages: ${missing[*]} and there is no apt-get here"
  if [ "$(id -u)" = "0" ]; then
    SUDO=()
  elif sudo -n true >/dev/null 2>&1; then
    SUDO=(sudo -n)
  else
    die "missing packages: ${missing[*]}. Install them (sudo apt-get install -y ${missing[*]}) and run this again"
  fi
  say "installing with apt: ${missing[*]}"
  # A freshly booted box often still holds the dpkg lock for its own updates: wait for it.
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 update -q
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 install -y -q "${missing[@]}"
fi

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || die "$PYTHON is $("$PYTHON" --version 2>&1). Foul Play needs Python 3.11 or newer (set PYTHON=...)"
"$PYTHON" -c "import venv, ensurepip" >/dev/null 2>&1 \
  || die "$PYTHON cannot create virtualenvs (python3-venv is missing)"
command -v cc >/dev/null 2>&1 || die "no C linker (cc). Rust needs one: install build-essential"

# ---- 2. the pinned checkout ---------------------------------------------------------------
if [ -d "$FP_DIR/.git" ]; then
  :
elif [ -e "$FP_DIR" ] && [ -n "$(ls -A "$FP_DIR" 2>/dev/null)" ]; then
  die "$FP_DIR exists, is not empty and is not a git checkout. Give another directory."
else
  mkdir -p "$FP_DIR"
  git -C "$FP_DIR" init -q
  git -C "$FP_DIR" remote add origin "$FOULPLAY_REPO"
fi
head_now="$(git -C "$FP_DIR" rev-parse -q --verify HEAD 2>/dev/null || true)"
if [ "$head_now" != "$COMMIT" ]; then
  if ! git -C "$FP_DIR" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
    say "fetching Foul Play $COMMIT"
    git -C "$FP_DIR" fetch -q --depth 1 origin "$COMMIT"
  fi
  git -C "$FP_DIR" checkout -q --detach "$COMMIT"
fi
[ "$(git -C "$FP_DIR" rev-parse HEAD)" = "$COMMIT" ] || die "$FP_DIR is not at $COMMIT"
[ -f "$FP_DIR/run.py" ] && [ -d "$FP_DIR/fp" ] || die "$FP_DIR does not look like Foul Play (no run.py or fp/)"
say "Foul Play $COMMIT in $FP_DIR"

REQ="$FP_DIR/requirements.txt"
ENGINE_VERSION="$(grep -oE '^poke-engine==[0-9]+\.[0-9]+\.[0-9]+' "$REQ" | head -n 1 | sed 's/^poke-engine==//')"
[ -n "$ENGINE_VERSION" ] || die "could not read the poke-engine version from $REQ"
grep -q "$ENGINE_FEATURES" "$REQ" \
  || say "WARNING: $REQ no longer asks for $ENGINE_FEATURES. Building with it anyway: gen9 random battles need it."

VENV="$FP_DIR/.venv"
PY="$VENV/bin/python"
STAMP="$VENV/psppo-install-stamp"
WANT="commit=$COMMIT poke-engine=$ENGINE_VERSION features=$ENGINE_FEATURES"

# ---- 6 (defined here, used below). does the installed engine work, with Tera? --------------
verify_engine() {
  ( cd "$FP_DIR" && "$PY" - <<'PYEOF'
import sys
import poke_engine
from poke_engine import Move, Pokemon, Side, State, monte_carlo_tree_search

import dateutil  # noqa: F401
import requests  # noqa: F401
import websockets  # noqa: F401


def mon(name, types, tera):
    return Pokemon(
        id=name, level=80, types=types, base_types=types, hp=250, maxhp=250,
        attack=200, defense=200, special_attack=200, special_defense=200, speed=200,
        moves=[Move(id="tackle", pp=32), Move(id="ember", pp=32)], tera_type=tera,
    )


state = State(
    side_one=Side(pokemon=[mon("charmander", ("fire", "typeless"), "water"), mon("squirtle", ("water", "typeless"), "fire")]),
    side_two=Side(pokemon=[mon("bulbasaur", ("grass", "typeless"), "fire"), mon("pikachu", ("electric", "typeless"), "grass")]),
)
result = monte_carlo_tree_search(State.from_string(state.to_string()), 100)
choices = [option.move_choice for option in result.side_one]
if result.total_visits <= 0:
    sys.exit("the search ran no iterations")
if not any(choice.endswith("-tera") for choice in choices):
    sys.exit("the engine offers no Tera move (%s): it was built without the terastallization feature" % choices)
print("engine ok: %d iterations in 100 ms, options %s" % (result.total_visits, choices))
PYEOF
  )
}

if [ -x "$PY" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ] && verify_engine >/dev/null 2>&1; then
  say "virtualenv and engine already installed ($WANT)"
else
  # ---- 3. Rust ----------------------------------------------------------------------------
  export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
  export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
  case ":$PATH:" in
    *":$CARGO_HOME/bin:"*) ;;
    *) PATH="$CARGO_HOME/bin:$PATH"; export PATH ;;
  esac
  if ! command -v cargo >/dev/null 2>&1; then
    say "installing Rust $RUST_TOOLCHAIN with rustup into $CARGO_HOME (no dotfile is changed)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN" --no-modify-path
  fi
  command -v cargo >/dev/null 2>&1 || die "cargo is still not on PATH after installing Rust"
  say "$(cargo --version)"
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"

  # ---- 4. virtualenv and requirements ------------------------------------------------------
  if [ ! -x "$PY" ]; then
    say "creating $VENV with $("$PYTHON" --version 2>&1)"
    "$PYTHON" -m venv "$VENV"
  fi
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  # Everything except the engine, exactly as Foul Play pins it.
  grep -vE '^[[:space:]]*poke-engine' "$REQ" > "$VENV/requirements-without-engine.txt"
  "$PY" -m pip install --quiet -r "$VENV/requirements-without-engine.txt"
  # The engine is on PyPI as source only. pip caches the wheel it builds and cannot tell which
  # features a cached wheel was built with, so a cache hit from another generation's build
  # would silently give an engine without Tera. Build it fresh, with the features spelled out.
  say "building poke-engine $ENGINE_VERSION with $ENGINE_FEATURES (a few minutes the first time)"
  "$PY" -m pip install --quiet --no-cache-dir --force-reinstall --no-deps \
    "poke-engine==$ENGINE_VERSION" \
    --config-settings="build-args=--features $ENGINE_FEATURES --no-default-features" \
    || die "building poke-engine failed with $(cargo --version). It was tested with Rust $PINNED_RUST: rustup toolchain install $PINNED_RUST && rustup default $PINNED_RUST, then run this again"
  verify_engine
  printf '%s\n' "$WANT" > "$STAMP"
fi

# ---- 5. set data --------------------------------------------------------------------------
# Foul Play downloads this in its first battle when the file is missing, and if that download
# returns an error page it caches an EMPTY set list and plays on without knowing any sets.
# Several processes starting together would also race on the file. Put it there now.
SETS_DIR="$FP_DIR/fp/data/pkmn_sets_cache"
SETS_FILE="$SETS_DIR/gen9randombattle.json"
mkdir -p "$SETS_DIR"
sets_ok() { [ -f "$SETS_FILE" ] && echo "$SETS_SHA256  $SETS_FILE" | sha256sum -c - >/dev/null 2>&1; }
if sets_ok; then
  say "set data already in place"
else
  if [ -f "$SETS_FILE" ]; then
    # Not the pinned file (an older download, or the empty one). Keep it under another name.
    mv "$SETS_FILE" "$SETS_FILE.replaced-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  say "downloading the gen9 random battle sets"
  curl -fsSL --retry 3 -o "$SETS_FILE.part" "$SETS_URL"
  echo "$SETS_SHA256  $SETS_FILE.part" | sha256sum -c - >/dev/null \
    || die "the downloaded set data does not match the pinned checksum ($SETS_FILE.part)"
  mv "$SETS_FILE.part" "$SETS_FILE"
fi
"$PY" - "$SETS_FILE" <<'PYEOF'
import json
import sys

sets = json.load(open(sys.argv[1]))
if len(sets) < 400:
    sys.exit("the set data has only %d species" % len(sets))
print("[install_foulplay] set data: %d species" % len(sets))
PYEOF

# ---- 6. final checks ----------------------------------------------------------------------
verify_engine | sed 's/^/[install_foulplay] /'
( cd "$FP_DIR" && "$PY" run.py --help >/dev/null ) || die "run.py --help failed"

say "done: foul-play $COMMIT at $FP_DIR"
if [ "$FP_DIR" != "$HOME/foul-play" ]; then
  say "this is not the default place: export FOULPLAY_DIR=$FP_DIR before running the benchmark"
fi
