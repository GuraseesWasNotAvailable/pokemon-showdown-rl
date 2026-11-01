#!/usr/bin/env bash
# Installs and enables the two systemd units on the training box. Run ON the box, as ubuntu,
# after cloud/bootstrap.sh:
#
#   bash ~/ps-ppo/cloud/install_services.sh             install and enable (they start at every boot)
#   bash ~/ps-ppo/cloud/install_services.sh --now       ... and start them now
#   bash ~/ps-ppo/cloud/install_services.sh --print     only print the units as they would be installed
#
#   psppo-train   pipeline.py: servers, trainer, the three phases, restarts
#   psppo-eval    evaluate.py --watch: win rates of new checkpoints
#
# The unit files in cloud/systemd/ are written for the box's layout:
#   /home/ubuntu/ps-ppo   /home/ubuntu/venv/bin/python   /home/ubuntu/runs/main   cloud/run_vm.json
# For another layout set REPO, VENV_PY, RUN_DIR, CONFIG, ENV_FILE or RUN_USER before calling:
# the paths are rewritten on the way in. Safe to run again after a redeploy.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
RUN_USER="${RUN_USER:-$(id -un)}"
# The group's name, or its number when it has none: "id -gn" then prints the number and fails, so
# take what it printed, first line only (a second line would break the sed expressions below).
RUN_GROUP="${RUN_GROUP:-$(id -gn 2>/dev/null || true)}"
RUN_GROUP="${RUN_GROUP%%$'\n'*}"
[ -n "$RUN_GROUP" ] || RUN_GROUP="$(id -g)"
USER_HOME="${USER_HOME:-$HOME}"
VENV_PY="${VENV_PY:-$USER_HOME/venv/bin/python}"
RUN_DIR="${RUN_DIR:-$USER_HOME/runs/main}"
CONFIG="${CONFIG:-$REPO/cloud/run_vm.json}"
ENV_FILE="${ENV_FILE:-$USER_HOME/psppo.env}"
UNITS="psppo-train.service psppo-eval.service"

NOW=0
PRINT=0
for arg in "$@"; do
  case "$arg" in
    --now) NOW=1 ;;
    --print) PRINT=1 ;;
    -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    *) echo "unknown argument: $arg"; sed -n 2,15p "$0"; exit 2 ;;
  esac
done

# The unit text with this box's paths. Two steps (layout path -> marker -> value) so a value that
# happens to contain one of the layout paths is not rewritten a second time.
render() {
  sed -e 's|/home/ubuntu/ps-ppo/cloud/run_vm.json|@@CONFIG@@|g' \
      -e 's|/home/ubuntu/venv/bin/python|@@PY@@|g' \
      -e 's|/home/ubuntu/runs/main|@@RUN@@|g' \
      -e 's|/home/ubuntu/psppo.env|@@ENV@@|g' \
      -e 's|/home/ubuntu/ps-ppo|@@REPO@@|g' \
      -e 's|^User=ubuntu$|User=@@USER@@|' \
      -e 's|^Group=ubuntu$|Group=@@GROUP@@|' \
      -e "s|@@CONFIG@@|$CONFIG|g" -e "s|@@PY@@|$VENV_PY|g" -e "s|@@RUN@@|$RUN_DIR|g" -e "s|@@ENV@@|$ENV_FILE|g" \
      -e "s|@@REPO@@|$REPO|g" -e "s|@@USER@@|$RUN_USER|" -e "s|@@GROUP@@|$RUN_GROUP|" \
      "$HERE/systemd/$1"
}

if [ "$PRINT" = 1 ]; then
  for u in $UNITS; do echo "##### /etc/systemd/system/$u"; render "$u"; echo; done
  exit 0
fi

echo "repo $REPO | python $VENV_PY | run directory $RUN_DIR | config $CONFIG | user $RUN_USER"
[ -x "$VENV_PY" ] || { echo "no Python at $VENV_PY (run cloud/bootstrap.sh first, or set VENV_PY)"; exit 1; }
[ -f "$REPO/pipeline.py" ] || { echo "no pipeline.py in $REPO"; exit 1; }
[ -f "$CONFIG" ] || { echo "no config at $CONFIG"; exit 1; }
command -v systemctl >/dev/null 2>&1 || { echo "systemctl not found: this is not a systemd machine"; exit 1; }
command -v node >/dev/null 2>&1 || echo "WARNING: node is not on PATH here. If it is not in /usr/bin, put a PATH= line in $ENV_FILE."
[ -f "$REPO/evaluate.py" ] || echo "WARNING: $REPO/evaluate.py is missing. psppo-eval will keep restarting until it is deployed."

mkdir -p "$RUN_DIR"
# A config that pipeline.py rejects must stop us here, not in a restart loop later.
"$VENV_PY" "$REPO/pipeline.py" --config "$CONFIG" --run-dir "$RUN_DIR" --dry-run >/dev/null \
  || { echo "pipeline.py rejects $CONFIG (see the message above). Nothing installed."; exit 1; }
# What this config does with the machine when the run ends, well or badly: shut it down, or leave
# it up and paid for with nothing running ("on_done", "on_failed"). Shown here, where it is read.
"$VENV_PY" "$REPO/pipeline.py" --config "$CONFIG" --run-dir "$RUN_DIR" --dry-run | sed -n '/^== when/,$p' || true
# The "eval" section is read by evaluate.py, which carries on past a key it does not know (it logs
# it once and uses its default). Print what it will do, and any such key, here where it is seen.
if [ -f "$REPO/evaluate.py" ]; then
  "$VENV_PY" "$REPO/evaluate.py" --run-dir "$RUN_DIR" --check-config "$CONFIG" \
    || echo "WARNING: evaluate.py found something wrong in the \"eval\" section of $CONFIG (see above). psppo-eval uses its defaults for those keys."
fi

for u in $UNITS; do
  render "$u" | sudo tee "/etc/systemd/system/$u" >/dev/null
  sudo chmod 0644 "/etc/systemd/system/$u"
  echo "installed /etc/systemd/system/$u"
done
sudo systemctl daemon-reload
if command -v systemd-analyze >/dev/null 2>&1; then
  for u in $UNITS; do sudo systemd-analyze verify "/etc/systemd/system/$u" 2>&1 | sed 's/^/  verify: /' || true; done
fi
# shellcheck disable=SC2086
sudo systemctl enable $UNITS

if [ "$NOW" = 1 ]; then
  # shellcheck disable=SC2086
  sudo systemctl start $UNITS
  sleep 3
fi
for u in $UNITS; do
  echo "$u: enabled=$(systemctl is-enabled "$u" 2>/dev/null || true) active=$(systemctl is-active "$u" 2>/dev/null || true)"
done
cat <<EOF

Both units start at every boot. Now:
  sudo systemctl start psppo-train psppo-eval         start (or resume) the run and the evaluator
  $VENV_PY $REPO/status.py --run-dir $RUN_DIR
  journalctl -u psppo-train -f                        the supervisor's log
  sudo systemctl stop psppo-train                     clean stop (saves a checkpoint; resumes on start)
A run that is done or marked failed is not started again:
  $VENV_PY $REPO/pipeline.py --run-dir $RUN_DIR --clear-failed    after fixing what made it fail
EOF
