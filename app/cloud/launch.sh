#!/usr/bin/env bash
# Launch the training box on AWS (us-east-1) as a persistent spot instance.
#
#   bash cloud/launch.sh                      # 64-vCPU L40S first, then the 32-vCPU one
#   bash cloud/launch.sh g6e.8xlarge:b        # one type in one zone (a, b, c or d)
#
# "Persistent" with "stop" means a spot interruption stops the box instead of destroying it:
# the disk and every checkpoint on it survive, AWS starts the box again when capacity returns,
# and the training service resumes from the newest checkpoint by itself.
#
# What it costs while it runs (spot, 2026-10-05): g6e.16xlarge about $2.10 to $2.40 an hour,
# g6e.8xlarge about $1.60 an hour, plus $0.55 a day for the 200 GB disk. To stop paying, run
# cloud/teardown.sh (it asks before it does anything).
#
# Needs, once: the key pair "psppo-key" (private half in ~/.ssh/psppo-key.pem) and the security
# group "psppo-sg" (SSH from this laptop's address only). Both already exist.
export MSYS_NO_PATHCONV=1
set -u
R=us-east-1
AMI=ami-0c244e62d32d4ca4e   # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261002
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="$HERE/.state"
mkdir -p "$STATE"
# aws.exe wants a Windows path after file://, Git Bash gives /c/...; cygpath fixes that there.
if command -v cygpath >/dev/null 2>&1; then WSTATE="$(cygpath -m "$STATE")"; else WSTATE="$STATE"; fi

SG=$(aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=psppo-sg \
      --query "SecurityGroups[0].GroupId" --output text)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then echo "security group psppo-sg not found"; exit 1; fi

RUNNING=$(aws ec2 describe-instances --region $R \
      --filters Name=tag:Name,Values=psppo-train Name=instance-state-name,Values=pending,running,stopping,stopped \
      --query "Reservations[].Instances[].[InstanceId,State.Name]" --output text)
if [ -n "$RUNNING" ]; then echo "a psppo-train box already exists: $RUNNING"; exit 1; fi

cat > "$STATE/bdm.json" <<'EOF'
[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Iops":3000,"Throughput":250,"DeleteOnTermination":true}}]
EOF
cat > "$STATE/market.json" <<'EOF'
{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}
EOF
cat > "$STATE/tags.json" <<'EOF'
[{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"psppo-train"},{"Key":"project","Value":"ps-ppo"}]},
 {"ResourceType":"volume","Tags":[{"Key":"Name","Value":"psppo-train"},{"Key":"project","Value":"ps-ppo"}]},
 {"ResourceType":"spot-instances-request","Tags":[{"Key":"Name","Value":"psppo-train"},{"Key":"project","Value":"ps-ppo"}]}]
EOF

subnet_of() {
  aws ec2 describe-subnets --region $R --filters Name=default-for-az,Values=true \
      Name=availability-zone,Values=${R}$1 --query "Subnets[0].SubnetId" --output text
}

if [ $# -eq 0 ]; then
  set -- g6e.16xlarge:a g6e.16xlarge:d g6e.16xlarge:c g6e.16xlarge:b \
         g6e.8xlarge:b g6e.8xlarge:a g6e.8xlarge:c g6e.8xlarge:d
fi

for combo in "$@"; do
  T=${combo%%:*}; Z=${combo##*:}
  SUBNET=$(subnet_of "$Z")
  echo ">>> trying $T in ${R}${Z}"
  OUT=$(aws ec2 run-instances --region $R --image-id $AMI --instance-type "$T" --key-name psppo-key \
      --security-group-ids "$SG" --subnet-id "$SUBNET" --count 1 \
      --block-device-mappings "file://$WSTATE/bdm.json" \
      --instance-market-options "file://$WSTATE/market.json" \
      --tag-specifications "file://$WSTATE/tags.json" \
      --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
      --instance-initiated-shutdown-behavior stop \
      --query "Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,SpotInstanceRequestId]" \
      --output text 2>&1)
  if echo "$OUT" | grep -q "^i-"; then
    echo "$OUT" > "$STATE/instance.txt"
    echo "LAUNCHED  $OUT"
    echo "Claude picks it up from here (cloud/.state/instance.txt)."
    exit 0
  fi
  echo "    no: $(echo "$OUT" | tail -1 | cut -c1-200)"
done
echo "No capacity in any combination tried. Run it again in a few minutes, or name another type."
exit 1
