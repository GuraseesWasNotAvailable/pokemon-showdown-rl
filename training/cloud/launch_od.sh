#!/usr/bin/env bash
# Launch an ON-DEMAND training box (no spot: it is never interrupted, and it costs the list price
# until it is stopped or terminated).
#
#   bash cloud/launch_od.sh <region> <instance-type> [zone letters, default: a b c]
#   AWSCMD="bash /path/to/wrapper.sh" bash cloud/launch_od.sh us-east-2 g6e.16xlarge
#
# AWSCMD is the command that runs the aws CLI for the account the box goes into (default: plain
# "aws", your default profile). The key pair "psppo-key" and the security group "psppo-sg" must
# already exist in that account and region.
set -u
AWSCMD=${AWSCMD:-aws}
R=${1:?region}; T=${2:?instance type}; shift 2
if [ $# -eq 0 ]; then set -- a b c; fi
NAME=${PSPPO_NAME:-psppo-train}
x() { MSYS_NO_PATHCONV=1 $AWSCMD "$@"; }
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="$HERE/.state"; mkdir -p "$STATE"
if command -v cygpath >/dev/null 2>&1; then WSTATE="$(cygpath -m "$STATE")"; else WSTATE="$STATE"; fi

# The image is public, so any credentials that can read the public parameter will do.
AMI=${PSPPO_AMI:-$(MSYS_NO_PATHCONV=1 aws ssm get-parameter --region "$R" \
      --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
      --query Parameter.Value --output text)}
SG=$(x ec2 describe-security-groups --region "$R" --filters Name=group-name,Values=psppo-sg \
      --query "SecurityGroups[0].GroupId" --output text)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then echo "security group psppo-sg not found in $R"; exit 1; fi
EXISTING=$(x ec2 describe-instances --region "$R" \
      --filters "Name=tag:Name,Values=$NAME" Name=instance-state-name,Values=pending,running,stopping,stopped \
      --query "Reservations[].Instances[].[InstanceId,State.Name]" --output text)
if [ -n "$EXISTING" ]; then echo "a $NAME box already exists in $R: $EXISTING"; exit 1; fi

cat > "$STATE/bdm_od.json" <<'EOF'
[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3","Iops":3000,"Throughput":250,"DeleteOnTermination":true}}]
EOF
cat > "$STATE/tags_od.json" <<EOF
[{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"$NAME"},{"Key":"project","Value":"ps-ppo"}]},
 {"ResourceType":"volume","Tags":[{"Key":"Name","Value":"$NAME"},{"Key":"project","Value":"ps-ppo"}]}]
EOF

for Z in "$@"; do
  SUBNET=$(x ec2 describe-subnets --region "$R" --filters Name=default-for-az,Values=true \
        "Name=availability-zone,Values=${R}${Z}" --query "Subnets[0].SubnetId" --output text)
  echo ">>> trying $T on demand in ${R}${Z}"
  OUT=$(x ec2 run-instances --region "$R" --image-id "$AMI" --instance-type "$T" --key-name psppo-key \
      --security-group-ids "$SG" --subnet-id "$SUBNET" --count 1 \
      --block-device-mappings "file://$WSTATE/bdm_od.json" \
      --tag-specifications "file://$WSTATE/tags_od.json" \
      --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
      --instance-initiated-shutdown-behavior stop \
      --query "Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]" --output text 2>&1)
  if echo "$OUT" | grep -q "^i-"; then
    echo "$OUT $R" > "$STATE/instance_od.txt"
    echo "LAUNCHED  $OUT"
    exit 0
  fi
  echo "    no: $(echo "$OUT" | tail -1 | cut -c1-240)"
done
echo "Not launched."
exit 1
