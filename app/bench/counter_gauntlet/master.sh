#!/usr/bin/env bash
# The counter gauntlet: mix against mix-counter, opponents in the owner's order (the set-up test bot, Foul Play
# at fair compute, the network alone), 150 games per arm each, no game started after BUDGET minutes from the
# start. One server on port 10500 for all three; it is stopped at the end. Touch $D/stop to stop after the
# current phase (and $D/g/<phase>/stop to stop the current phase's new games). A phase that ends without a
# single result stops the master (the next phase does not start in its place).
D=$HOME/dev/counter-gauntlet
PORT=10500
BUDGET=${BUDGET:-95}
T0=$(date +%s)
left() { echo $(( BUDGET - ($(date +%s) - T0) / 60 )); }
ok() { [ -s "$D/g/$1/results.jsonl" ] || { echo "$(date -u +%FT%TZ) phase $1 has no results: master stops" >> $D/master.log; touch $D/stop; }; }
echo "$(date -u +%FT%TZ) master start, budget $BUDGET min" >> $D/master.log
$D/server.sh $PORT >> $D/master.log 2>&1
# 1. the set-up test bot: parallel 7 on 8 helpers (11 processes with the server and the bot), at most 26 minutes
L=$(left); M=$(( L < 26 ? L : 26 ))
[ -e $D/stop ] || { $D/phase.sh su setup 7 8 $M 150 $PORT 450 2 41 >> $D/master.log 2>&1; ok su; }
# 2. Foul Play at fair compute (450 ms on 3 helpers of its own): parallel 5 on 6 helpers (12 processes); about
#    30 minutes are kept for the network alone
L=$(left); M=$(( L - 30 )); [ $M -lt 15 ] && M=15; [ $M -gt $L ] && M=$L
[ -e $D/stop ] || [ $L -lt 3 ] || { $D/phase.sh fp foulplay 5 6 $M 150 $PORT 450 3 42 >> $D/master.log 2>&1; ok fp; }
# 3. the network alone (011850): parallel 7 on 8 helpers, until the budget ends
L=$(left)
[ -e $D/stop ] || [ $L -lt 3 ] || { $D/phase.sh nt net 7 8 $L 150 $PORT 450 2 43 >> $D/master.log 2>&1; ok nt; }
P=$(awk '{print $2}' $D/srv$PORT.pgid)
kill -- -$P 2>/dev/null
sleep 5
echo "$(date -u +%FT%TZ) server group $P stopped; master all phases done ($(( ($(date +%s) - T0) / 60 )) min)" >> $D/master.log
