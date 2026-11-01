#!/usr/bin/env bash
# Talk to the training box from the laptop.
#
#   bash cloud/vm.sh info                 instance id, type, state, address, spot price now
#   bash cloud/vm.sh ssh [command...]     a shell on the box, or run one command there
#   bash cloud/vm.sh tty command          one command on the box with a terminal (ssh -t), e.g. tmux
#   bash cloud/vm.sh deploy [--head]      copy this working tree (or only what is committed) to ~/ps-ppo
#   bash cloud/vm.sh release <name>       copy this working tree to ~/rel/<name> on the box, beside
#                                         the code that is running (nothing running is touched)
#   bash cloud/vm.sh activate <name>      make ~/ps-ppo point at ~/rel/<name>. Only with the
#                                         trainer and evaluator stopped
#   bash cloud/vm.sh status               the run's status screen (status.py on the box)
#   bash cloud/vm.sh pull [name]          copy the newest checkpoint (or the named one) and the
#                                         evaluation results to runs/vm/ on the laptop
#   bash cloud/vm.sh allow-ip             let this laptop's current address in (run it when SSH
#                                         stops answering after a network change)
#
# The box's address changes whenever it stops and starts (spot interruptions), so it is looked
# up from AWS every time, by the tag Name=psppo-train.
# Git Bash rewrites arguments that look like paths. aws.exe must not get that (it breaks
# things like SSM parameter names), git.exe needs it, so it is switched off for aws only.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# Which box. There can be more than one (for example the first spot box in one account and an
# on-demand box in another). Each has a file cloud/.state/targets/<name>.env that may set
# PSPPO_REGION, PSPPO_KEY, PSPPO_NAME (the Name tag) and AWSCMD (the command that runs the aws
# CLI for that account). cloud/.state/current holds the name used by default, and
# PSPPO_TARGET=<name> picks another for one call. With no files at all, the first box.
TARGET=${PSPPO_TARGET:-$(cat "$HERE/.state/current" 2>/dev/null || echo a)}
if [ -f "$HERE/.state/targets/$TARGET.env" ]; then . "$HERE/.state/targets/$TARGET.env"; fi
R=${PSPPO_REGION:-us-east-1}
KEY=${PSPPO_KEY:-$HOME/.ssh/psppo-key.pem}
NAME=${PSPPO_NAME:-psppo-train}
AWSCMD=${AWSCMD:-aws}
aws() { MSYS_NO_PATHCONV=1 command $AWSCMD "$@"; }
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/psppo_known_hosts"
          -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=15 -o LogLevel=ERROR)

describe() {
  aws ec2 describe-instances --region $R \
    --filters "Name=tag:Name,Values=$NAME" Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress,Placement.AvailabilityZone,SpotInstanceRequestId]" \
    --output text
}

ip() {
  local a
  a=$(describe | awk '$3=="running"{print $4; exit}')
  if [ -z "$a" ] || [ "$a" = "None" ]; then echo "no running $NAME box for target $TARGET in $R (state: $(describe | awk '{print $3}' | head -1))" >&2; return 1; fi
  echo "$a"
}

cmd=${1:-info}; shift || true
case "$cmd" in
  info)
    D=$(describe)
    if [ -z "$D" ]; then echo "no $NAME box exists for target $TARGET in $R"; exit 1; fi
    echo "$D"
    T=$(echo "$D" | awk '{print $2; exit}'); Z=$(echo "$D" | awk '{print $5; exit}')
    P=$(aws ec2 describe-spot-price-history --region $R --instance-types "$T" --availability-zone "$Z" \
          --product-descriptions "Linux/UNIX" --max-items 1 --query "SpotPriceHistory[0].SpotPrice" --output text 2>/dev/null | head -1)
    echo "spot price now: \$$P an hour"
    ;;
  ip) ip ;;
  ssh)
    A=$(ip) || exit 1
    if [ $# -eq 0 ]; then exec ssh "${SSH_OPTS[@]}" "ubuntu@$A"; fi
    exec ssh "${SSH_OPTS[@]}" "ubuntu@$A" "$@"
    ;;
  tty)
    # One command on the box with a terminal of its own (ssh -t), for anything that is typed
    # into or attached to, like tmux. PSPPO_SSH_TTY=-tt forces the terminal when this end has
    # none (a test that types through a pipe).
    A=$(ip) || exit 1
    exec ssh ${PSPPO_SSH_TTY:--t} "${SSH_OPTS[@]}" "ubuntu@$A" "$@"
    ;;
  release)
    # Every trainer start imports the files as they are on disk at that moment, and the
    # supervisor restarts the trainer by itself. So new code never goes over the running code:
    # it goes into its own directory, and ~/ps-ppo (a link) is moved to it between runs.
    REL=${1:?release name}
    A=$(ip) || exit 1
    COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
    DIRTY=$(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "test ! -e ~/rel/$REL && mkdir -p ~/rel/$REL" || { echo "~/rel/$REL already exists on the box: pick another name"; exit 1; }
    (cd "$REPO" && git ls-files -co --exclude-standard -z | tar --null -T - -czf - ) \
      | ssh "${SSH_OPTS[@]}" "ubuntu@$A" "tar -xzf - -C ~/rel/$REL" || exit 1
    (cd "$REPO" && tar --exclude='heuristic/cache/top' -czf - heuristic/cache) \
      | ssh "${SSH_OPTS[@]}" "ubuntu@$A" "tar -xzf - -C ~/rel/$REL" || exit 1
    # The old code the speed tests compare with (bench/micro_ref.py checks every byte of it).
    REFTMP=$(mktemp -d) || exit 1
    for m in utils config ppo_core inference learner obs_assembler obs_moves obs_pokemon obs_abilities obs_global obs_transitions; do
      git -C "$REPO" show "b5680bd:$m.py" > "$REFTMP/$m.py" || { echo "commit b5680bd is not in this clone"; exit 1; }
    done
    tar -czf - -C "$REFTMP" . | ssh "${SSH_OPTS[@]}" "ubuntu@$A" "mkdir -p ~/runs/ref_b5680bd && tar -xzf - -C ~/runs/ref_b5680bd" || exit 1
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "cd ~/rel/$REL && echo '$REL $COMMIT +$DIRTY $(date -u +%Y-%m-%dT%H:%M:%SZ)' > DEPLOYED && chmod +x cloud/*.sh bench/*.sh 2>/dev/null; cat DEPLOYED"
    ;;
  activate)
    REL=${1:?release name}
    A=$(ip) || exit 1
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "set -e; test -f ~/rel/$REL/train.py || { echo 'no such release on the box'; exit 1; }
      if systemctl is-active --quiet psppo-train 2>/dev/null || systemctl is-active --quiet psppo-eval 2>/dev/null || pgrep -f '[t]rain.py --mode' >/dev/null 2>&1; then echo 'a trainer or the evaluator is running: stop it first'; exit 1; fi
      if [ -d ~/ps-ppo ] && [ ! -L ~/ps-ppo ]; then mv ~/ps-ppo ~/rel/first_\$(date -u +%Y%m%d_%H%M%S); fi
      ln -sfn ~/rel/$REL ~/ps-ppo; echo \"~/ps-ppo -> \$(readlink ~/ps-ppo)\"; cat ~/ps-ppo/DEPLOYED"
    ;;
  deploy)
    A=$(ip) || exit 1
    # Copying over the files of a running trainer can leave it half old and half new.
    BUSY=$(ssh "${SSH_OPTS[@]}" "ubuntu@$A" "systemctl is-active --quiet psppo-train 2>/dev/null && echo psppo-train; pgrep -f '[t]rain.py --mode' >/dev/null 2>&1 && echo train.py; true")
    if [ -n "$BUSY" ] && [ "${PSPPO_DEPLOY_OVER_RUNNING:-0}" != 1 ]; then
      echo "not deploying to $A: $(echo $BUSY) is running there. Use 'release' and 'activate', or stop it first." >&2
      exit 1
    fi
    COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
    DIRTY=$(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')
    echo "deploying $COMMIT (+$DIRTY uncommitted paths) to $A"
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "mkdir -p ~/ps-ppo ~/runs/main" || exit 1
    if [ "${1:-}" = "--head" ]; then
      # Only what is committed: use this while files in the working tree are half-written.
      DIRTY="0 (committed files only)"
      git -C "$REPO" archive --format=tar.gz HEAD \
        | ssh "${SSH_OPTS[@]}" "ubuntu@$A" "tar -xzf - -C ~/ps-ppo" || exit 1
    else
      # Tracked files plus new ones that are not ignored, exactly as they are on disk now.
      (cd "$REPO" && git ls-files -co --exclude-standard -z | tar --null -T - -czf - ) \
        | ssh "${SSH_OPTS[@]}" "ubuntu@$A" "tar -xzf - -C ~/ps-ppo" || exit 1
    fi
    # The PSBot engine's data (ignored by git): send the laptop's copy so the teacher on the box
    # is byte for byte the one that was tested here. Not the replay archive under top/.
    (cd "$REPO" && tar --exclude='heuristic/cache/top' -czf - heuristic/cache) \
      | ssh "${SSH_OPTS[@]}" "ubuntu@$A" "tar -xzf - -C ~/ps-ppo" || exit 1
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "cd ~/ps-ppo && echo '$COMMIT +$DIRTY $(date -u +%Y-%m-%dT%H:%M:%SZ)' > DEPLOYED && chmod +x cloud/*.sh bench/*.sh 2>/dev/null; cat DEPLOYED"
    ;;
  status)
    A=$(ip) || exit 1
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "cd ~/ps-ppo && ~/venv/bin/python status.py --run-dir ~/runs/main $*"
    ;;
  pull)
    A=$(ip) || exit 1
    mkdir -p "$REPO/runs/vm/checkpoints" "$REPO/runs/vm/eval"
    CK=${1:-}
    if [ -z "$CK" ]; then CK=$(ssh "${SSH_OPTS[@]}" "ubuntu@$A" "ls -t ~/runs/main/checkpoints/*.pt 2>/dev/null | head -1 | xargs -r basename"); fi
    if [ -z "$CK" ]; then echo "no checkpoint on the box yet"; exit 1; fi
    scp "${SSH_OPTS[@]}" "ubuntu@$A:runs/main/checkpoints/$CK" "$REPO/runs/vm/checkpoints/$CK" || exit 1
    for f in eval/results.jsonl eval/REPORT.md state.json metrics.jsonl events.jsonl; do
      scp "${SSH_OPTS[@]}" "ubuntu@$A:runs/main/$f" "$REPO/runs/vm/$f" 2>/dev/null
    done
    ls -la "$REPO/runs/vm/checkpoints/$CK"
    ;;
  allow-ip)
    ME=$(curl -s --max-time 8 https://checkip.amazonaws.com | tr -d '[:space:]')
    SG=$(aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=psppo-sg --query "SecurityGroups[0].GroupId" --output text)
    aws ec2 authorize-security-group-ingress --region $R --group-id "$SG" --protocol tcp --port 22 --cidr "$ME/32" --query Return --output text 2>&1 | tail -1
    echo "SSH allowed from $ME"
    ;;
  *) echo "unknown command: $cmd"; sed -n 2,12p "$0"; exit 1 ;;
esac
