#!/usr/bin/env bash
# Go to the Ant — a short guided tour of all six emergent swarms. Pure Python, no dependencies.
set -e
PY=${PYTHON:-python3}
pause(){ echo; read -rp "  [enter] for the next swarm... " _ || true; echo; }

echo "════════ 1/6 · Ant foraging — a nest↔food trail self-assembles (minimum spanning tree) ════════"
$PY go_to_the_ant.py --ticks 3500 --ants 100 --seed 1; pause
echo "════════ 2/6 · Brood sorting — three item types self-cluster, no ant sorts ════════"
$PY brood_sorting.py --ticks 120000 --ants 45 --seed 0; pause
echo "════════ 3/6 · Termites — scattered deposits self-concentrate into columns ════════"
$PY termites.py --ticks 32000 --termites 90 --seed 4; pause
echo "════════ 4/6 · Wasps — identical wasps split into Chief / Foragers / Nurses ════════"
$PY wasps.py --ticks 5000 --wasps 80 --seed 0; pause
echo "════════ 5/6 · Flocking — random headings cohere into one banking flock ════════"
$PY flocking.py --ticks 700 --birds 90 --seed 1; pause
echo "════════ 6/6 · Wolves — six wolves surround the moose with no communication ════════"
$PY wolves.py --ticks 300 --wolves 6 --seed 3
echo
echo "For the LIVE interactive version of all six, open visualizer.html in any browser."
