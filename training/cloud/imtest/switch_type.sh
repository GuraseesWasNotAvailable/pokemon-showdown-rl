#!/usr/bin/env bash
# Runs ON THE LAPTOP (it needs the second account's keys, which never go to the box).
# Moves the training box to a faster instance type when one has capacity, without losing the run:
#   bash cloud/imtest/switch_type.sh p5.48xlarge [g6e.48xlarge ...]     try each type in order
#   bash cloud/imtest/switch_type.sh --probe p5.48xlarge g6e.48xlarge     only say whether one has capacity
# 1. Reserves one of the type in the box's zone (an open On-Demand Capacity Reservation, 3 hours):
#    capacity is then certain before anything is stopped. No capacity: nothing is touched, exit 1.
# 2. Stops the full run's units cleanly (pipeline.py saves a checkpoint), stops the box, sets the type,
#    starts it (it lands in the reservation), cancels the reservation (the box keeps running).
#    The units are enabled, so the run resumes from its checkpoint on boot.
# 3. If the start fails anyway, the type goes back to the one before and the box is started again.
set -uo pipefail
A="bash C:/Users/you/ps-ppo/cloud/.state/aws2.sh"
export MSYS_NO_PATHCONV=1
R=us-east-1; Z=us-east-1c; I=i-<instance-id>
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
probe=0
if [ "${1:-}" = "--probe" ]; then probe=1; shift; fi
[ $# -ge 1 ] || { echo "usage: $0 [--probe] TYPE [TYPE ...]"; exit 2; }
old=$($A ec2 describe-instances --region $R --instance-ids $I --query "Reservations[].Instances[].InstanceType" --output text)
cr=""; T=""
for t in "$@"; do
  [ "$t" = "$old" ] && { echo "already $t"; exit 0; }
  end=$(date -u -d "+180 minutes" +%Y-%m-%dT%H:%M:%SZ)
  out=$($A ec2 create-capacity-reservation --region $R --availability-zone $Z --instance-type "$t" --instance-platform Linux/UNIX \
        --instance-count 1 --end-date-type limited --end-date "$end" --instance-match-criteria open \
        --tag-specifications "ResourceType=capacity-reservation,Tags=[{Key=Name,Value=psppo-train-switch}]" \
        --query "CapacityReservation.CapacityReservationId" --output text 2>&1)
  if [[ "$out" == cr-* ]]; then cr=$out; T=$t; echo "reserved $t: $cr"; break; fi
  echo "$t: no capacity ($(echo "$out" | tail -1 | cut -c1-120))"
done
[ -n "$cr" ] || exit 1
if [ $probe = 1 ]; then
  $A ec2 cancel-capacity-reservation --region $R --capacity-reservation-id "$cr" --output text >/dev/null && echo "probe only: cancelled $cr"
  exit 0
fi
cd "$HERE"
echo "stopping the run's units cleanly"
bash cloud/vm.sh ssh 'sudo systemctl stop psppo-full-eval psppo-full-train; systemctl is-active psppo-full-train psppo-full-eval; tail -2 ~/runs/full/events.jsonl' || true
$A ec2 stop-instances --region $R --instance-ids $I --output text >/dev/null
$A ec2 wait instance-stopped --region $R --instance-ids $I && echo "stopped"
$A ec2 modify-instance-attribute --region $R --instance-id $I --instance-type Value="$T" && echo "type set to $T"
if $A ec2 start-instances --region $R --instance-ids $I --output text >/dev/null 2>&1 && $A ec2 wait instance-running --region $R --instance-ids $I; then
  echo "running as $T"
else
  echo "start as $T failed: back to $old"
  $A ec2 modify-instance-attribute --region $R --instance-id $I --instance-type Value="$old"
  $A ec2 start-instances --region $R --instance-ids $I --output text >/dev/null && $A ec2 wait instance-running --region $R --instance-ids $I && echo "running as $old again"
fi
$A ec2 cancel-capacity-reservation --region $R --capacity-reservation-id "$cr" --output text >/dev/null && echo "reservation $cr cancelled (the box keeps running)"
$A ec2 describe-instances --region $R --instance-ids $I --query "Reservations[].Instances[].[InstanceType,State.Name]" --output text
