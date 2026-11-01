#!/usr/bin/env bash
# Ends the AWS side of one box: cancels its persistent spot request (if it has one) and
# terminates the instance.
#
#   bash cloud/teardown.sh                        the box of the selected target
#   bash cloud/teardown.sh i-0123...              this one, when the target has more than one
#   PSPPO_TARGET=b bash cloud/teardown.sh         the box of target b
#
# WHICH BOX. There can be more than one (a spot box in one account, an on-demand box in another).
# This script looks at ONE target, chosen exactly as cloud/vm.sh chooses it: PSPPO_TARGET, else
# the name in cloud/.state/current, else "a". The target's file cloud/.state/targets/<name>.env
# gives the region (PSPPO_REGION), the Name tag (PSPPO_NAME) and the command that runs the aws CLI
# for that account (AWSCMD). The first thing printed is the target, the AWS account and the
# region, and every OTHER target on file is named with the command that looks at it: a box of
# another target is not seen from here and keeps costing money until it is torn down too.
#
# It shows the instance and its spot request, says what it is about to do, and then waits for you
# to TYPE THE INSTANCE ID. Anything else, or nothing, and it changes nothing. A spot request is
# cancelled first: a persistent request whose instance is terminated would otherwise launch a new
# box. An on-demand box has no request: nothing ever stops it but this.
#
# THE DISK GOES WITH THE INSTANCE (DeleteOnTermination in cloud/launch.sh and launch_od.sh): every
# checkpoint and the whole run directory. Copy what you want to keep first:
#   bash cloud/vm.sh pull
export MSYS_NO_PATHCONV=1
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="${PSPPO_STATE_DIR:-$HERE/.state}"

TARGET_FROM="the default"
TARGET="a"
if [ -n "${PSPPO_TARGET:-}" ]; then
  TARGET="$PSPPO_TARGET"; TARGET_FROM="PSPPO_TARGET"
elif [ -s "$STATE/current" ]; then
  TARGET="$(tr -d '[:space:]' < "$STATE/current")"; TARGET_FROM="$STATE/current"
fi
TARGET_FILE="$STATE/targets/$TARGET.env"
TARGET_MISSING=0
if [ -f "$TARGET_FILE" ]; then
  # shellcheck disable=SC1090
  . "$TARGET_FILE"
elif [ -d "$STATE/targets" ] && [ -n "$(ls "$STATE/targets" 2>/dev/null)" ]; then
  # Targets are on file but not this one: falling back to the defaults would show the first box
  # under the name of another.
  TARGET_MISSING=1
fi
R="${PSPPO_REGION:-us-east-1}"
TAG="${PSPPO_NAME:-psppo-train}"
# AWSCMD may be several words (a wrapper script with its interpreter), so it is run unquoted.
# AWS=... is the older name of the same setting.
AWSCMD="${AWSCMD:-${AWS:-aws}}"
aws_cli() {
  # shellcheck disable=SC2086
  command $AWSCMD "$@"
}

# confirm_typed EXPECTED WHAT: reads one line. Succeeds only if it is exactly EXPECTED.
confirm_typed() {
  local expected="$1" what="$2" answer=""
  printf 'Type the %s (%s) to go ahead. Anything else leaves everything as it is: ' "$what" "$expected"
  IFS= read -r answer || answer=""
  [ -n "$expected" ] && [ "$answer" = "$expected" ]
}

# The names of the other targets on file, space separated.
other_targets() {
  local f n out=""
  for f in "$STATE"/targets/*.env; do
    [ -f "$f" ] || continue
    n="$(basename "$f" .env)"
    [ "$n" = "$TARGET" ] || out="$out $n"
  done
  # shellcheck disable=SC2086,SC2116
  echo $out
}

remind_other_targets() {
  local others n
  others="$(other_targets)"
  [ -n "$others" ] || return 0
  echo
  echo "OTHER TARGETS ON FILE, not looked at by this call. Each may have a box that costs money:"
  for n in $others; do
    echo "    $n    PSPPO_TARGET=$n bash cloud/teardown.sh"
  done
}

# Spot requests with our tag that could still start or keep a box.
tagged_requests() {
  aws_cli ec2 describe-spot-instance-requests --region "$R" \
      --filters "Name=tag:Name,Values=$TAG" "Name=state,Values=open,active,disabled" \
      --query "SpotInstanceRequests[].[SpotInstanceRequestId,State,Status.Code,Type,InstanceId]" --output text
}

cancel_requests() {   # cancel_requests ID...  -> 0 only if AWS accepted the cancellation
  [ $# -gt 0 ] || return 0
  echo "cancelling spot request(s): $*"
  aws_cli ec2 cancel-spot-instance-requests --region "$R" --spot-instance-request-ids "$@" \
      --query "CancelledSpotInstanceRequests[].[SpotInstanceRequestId,State]" --output text
}

main() {
  local want="${1:-}" rows count line iid itype state az sir launched addr requests req_ids other r account alias market

  if [ "$TARGET_MISSING" = 1 ]; then
    echo "Target \"$TARGET\" (from $TARGET_FROM) has no file $TARGET_FILE."
    echo "The targets on file: $(other_targets)"
    echo "Name one of them:  PSPPO_TARGET=<name> bash cloud/teardown.sh     Nothing was changed."
    return 1
  fi

  account=$(aws_cli sts get-caller-identity --query Account --output text) \
    || { echo "Could not ask AWS which account \"$AWSCMD\" is logged in to (target $TARGET). Nothing was changed."; return 1; }
  alias=$(aws_cli iam list-account-aliases --query "AccountAliases[0]" --output text 2>/dev/null) || alias=""
  [ "$alias" = "None" ] && alias=""
  echo "Target    $TARGET   (chosen by $TARGET_FROM)"
  echo "Account   $account${alias:+   alias $alias}"
  echo "Region    $R   instances tagged Name=$TAG"
  echo "AWS CLI   $AWSCMD"
  echo

  rows=$(aws_cli ec2 describe-instances --region "$R" \
      --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,SpotInstanceRequestId,LaunchTime,PublicIpAddress]" \
      --output text) || { echo "Could not ask AWS for the instance (is the aws CLI installed and logged in?). Nothing was changed."; return 1; }
  requests=$(tagged_requests) || { echo "Could not ask AWS for the spot requests. Nothing was changed."; return 1; }

  if [ -z "$rows" ]; then
    echo "No $TAG instance exists in $R in account $account (pending, running, stopping or stopped)."
    if [ -z "$requests" ]; then
      echo "No open spot request with that tag either. Nothing to tear down for target $TARGET."
      remind_other_targets
      return 0
    fi
    echo "But these spot requests are still there and could launch a box:"
    echo "$requests" | sed 's/^/    /'
    req_ids=$(echo "$requests" | awk '{print $1}' | tr '\n' ' ')
    r=$(echo "$requests" | awk 'NR==1{print $1}')
    echo
    echo "This will cancel: $req_ids"
    confirm_typed "$r" "first spot request id" || { echo "Not confirmed. Nothing was changed."; remind_other_targets; return 1; }
    # shellcheck disable=SC2086
    cancel_requests $req_ids || { echo "AWS did not accept the cancellation. Check the console."; return 1; }
    echo "Done."
    remind_other_targets
    return 0
  fi

  count=$(echo "$rows" | wc -l | tr -d ' ')
  if [ -n "$want" ]; then
    line=$(echo "$rows" | awk -v id="$want" '$1==id')
    if [ -z "$line" ]; then
      echo "$want is not a $TAG instance in $R in account $account (target $TARGET). The ones that exist:"
      echo "$rows" | sed 's/^/    /'
      echo "Nothing was changed."
      remind_other_targets
      return 1
    fi
  elif [ "$count" -gt 1 ]; then
    echo "$count $TAG instances exist. Name the one to tear down:  bash cloud/teardown.sh <instance id>"
    echo "$rows" | sed 's/^/    /'
    echo "Nothing was changed."
    remind_other_targets
    return 1
  else
    line="$rows"
  fi

  iid=$(echo "$line" | awk '{print $1}')
  itype=$(echo "$line" | awk '{print $2}')
  state=$(echo "$line" | awk '{print $3}')
  az=$(echo "$line" | awk '{print $4}')
  sir=$(echo "$line" | awk '{print $5}')
  launched=$(echo "$line" | awk '{print $6}')
  addr=$(echo "$line" | awk '{print $7}')
  case "$iid" in i-*) ;; *) echo "Unexpected answer from AWS ($line). Nothing was changed."; return 1 ;; esac
  [ "$sir" = "None" ] && sir=""
  if [ -n "$sir" ]; then market="spot"; else market="ON DEMAND (list price, never interrupted)"; fi

  echo "Instance"
  echo "    id        $iid"
  echo "    type      $itype in $az, $market"
  echo "    state     $state"
  echo "    launched  $launched"
  echo "    address   $addr"
  echo "    target    $TARGET, account $account${alias:+ ($alias)}, region $R"
  if [ -n "$sir" ]; then
    echo "Spot request of this instance"
    aws_cli ec2 describe-spot-instance-requests --region "$R" --spot-instance-request-ids "$sir" \
        --query "SpotInstanceRequests[].[SpotInstanceRequestId,State,Status.Code,Type]" --output text | sed 's/^/    /'
  else
    echo "Spot request of this instance: none (not a spot instance)"
  fi
  # every tagged request that is still live, the instance's own first
  req_ids="$sir"
  other=$(echo "$requests" | awk -v own="$sir" 'NF && $1!=own {print $1}' | tr '\n' ' ')
  if [ -n "${other// /}" ]; then
    echo "Other live spot requests with the tag Name=$TAG"
    echo "$requests" | awk -v own="$sir" 'NF && $1!=own' | sed 's/^/    /'
    req_ids="$req_ids $other"
  fi
  # shellcheck disable=SC2086,SC2116
  req_ids=$(echo $req_ids)      # single spaces, no leading or trailing one

  echo
  echo "This will, in this order:"
  if [ -n "${req_ids// /}" ]; then
    echo "    1. cancel the spot request(s) $req_ids (so AWS does not start a replacement)"
    echo "    2. terminate instance $iid ($itype, target $TARGET, account $account${alias:+ $alias}, $R)"
  else
    echo "    1. terminate instance $iid ($itype, target $TARGET, account $account${alias:+ $alias}, $R)"
  fi
  echo "The instance's disk is deleted with it: every checkpoint, log and result on the box."
  echo "It cannot be undone. Copy what you need first (PSPPO_TARGET=$TARGET bash cloud/vm.sh pull)."
  echo
  confirm_typed "$iid" "instance id" || { echo "Not confirmed. Nothing was changed."; remind_other_targets; return 1; }

  if [ -n "${req_ids// /}" ]; then
    # shellcheck disable=SC2086
    cancel_requests $req_ids || {
      echo "AWS did not accept the cancellation, so the instance was NOT terminated"
      echo "(a persistent request would replace it). Check the console."
      return 1
    }
  fi
  echo "terminating $iid"
  aws_cli ec2 terminate-instances --region "$R" --instance-ids "$iid" \
      --query "TerminatingInstances[].[InstanceId,PreviousState.Name,CurrentState.Name]" --output text \
    || { echo "AWS did not accept the termination. The spot request is already cancelled; terminate $iid in the console."; return 1; }
  echo "Done. The instance stops costing money once its state is \"terminated\" (a minute or two):"
  echo "    $AWSCMD ec2 describe-instances --region $R --instance-ids $iid --query \"Reservations[].Instances[].State.Name\" --output text"
  echo "The files under cloud/.state/ (instance.txt, instance_od.txt) may still name this instance; the launch scripts write new ones."
  remind_other_targets
  return 0
}

# Only when run, not when sourced (sourcing just defines the functions above).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
