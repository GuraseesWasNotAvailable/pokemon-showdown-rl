#!/bin/bash
# boxssh.sh TARGET -- print the ssh command for a box of MY account (vm.sh target a, c or d).
ADDR=$(cd /c/Users/you/ps-ppo-v2 && PSPPO_TARGET=$1 bash cloud/vm.sh info 2>/dev/null | awk '$3=="running"{print $4; exit}')
[ -n "$ADDR" ] || { echo "no running box for target $1" >&2; exit 1; }
echo "ssh -i $HOME/.ssh/psppo-key.pem -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/psppo_known_hosts -o ConnectTimeout=15 -o ServerAliveInterval=30 -o LogLevel=ERROR ubuntu@$ADDR"
