#!/usr/bin/env bash
# After master.sh: the Foul Play phase (the first try stopped at once: --helpers must exceed --parallel), on port 10300 again.
D=$HOME/dev/setup-prove
for _ in $(seq 1 3600); do grep -q "master all phases done" $D/master.log && break; sleep 2; done
echo "$(date -u +%FT%TZ) master2 start" >> $D/master.log
$D/server.sh 10300 >> $D/master.log 2>&1
$D/phase.sh fp2 foulplay mix,mix-fixdefer 3 4 20 10300 450 2 34 0 >> $D/master.log 2>&1
P=$(awk "{print \$2}" $D/srv10300.pgid)
kill -- -$P 2>/dev/null
sleep 5
echo "$(date -u +%FT%TZ) server group $P stopped; master2 all phases done" >> $D/master.log
