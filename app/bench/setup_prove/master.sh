#!/usr/bin/env bash
# The set-up proof gauntlets, one after another on my own server of port 10300; the server is stopped at the end.
D=$HOME/dev/setup-prove
echo "$(date -u +%FT%TZ) master start" >> $D/master.log
$D/server.sh 10300 >> $D/master.log 2>&1
$D/phase.sh fp foulplay mix,mix-fixdefer 3 3 22 10300 450 3 31 0 >> $D/master.log 2>&1
$D/phase.sh nt net mix,mix-fixdefer 5 6 13 10300 450 2 32 0 >> $D/master.log 2>&1
$D/phase.sh su setup mix,mix-fixdefer 5 6 8 10300 450 2 33 0 >> $D/master.log 2>&1
P=$(awk "{print \$2}" $D/srv10300.pgid)
kill -- -$P 2>/dev/null
sleep 5
echo "$(date -u +%FT%TZ) server group $P stopped; master all phases done" >> $D/master.log
