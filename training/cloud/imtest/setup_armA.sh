#!/bin/bash
# Arm A of the imitation test: the network copies PSBot from random weights, on the baseline
# release (search 73f524b + GPU split + r2 worker fix), 25.4M decisions as the old run's imitation,
# then stops (warmup and ppo skipped; the full run is decided after the test).
set -euo pipefail
REL=/home/ubuntu/rel/baseline-c988471
RUN=/home/ubuntu/runs/armA
mkdir -p "$RUN/checkpoints"
if ls "$RUN/checkpoints"/*.pt >/dev/null 2>&1; then echo "armA already has checkpoints: not touching it"; exit 1; fi
/home/ubuntu/venv/bin/python - <<'EOF'
import json
src = json.load(open("/home/ubuntu/runs/baseline/run.json"))
cfg = {k: v for k, v in src.items() if not k.startswith("_")}
cfg["repo_dir"] = "/home/ubuntu/rel/baseline-c988471"
cfg["phases"] = {
    "imitation": {"skip": False, "min_decisions": 25_400_000, "max_decisions": 25_400_000,
                  "train": {"save_every": 25}, "extra_train_args": []},
    "warmup": {"skip": True, "min_decisions": 500_000, "max_decisions": 1_500_000, "extra_train_args": []},
    "ppo": {"skip": True, "target_decisions": 0, "extra_train_args": []},
}
cfg["on_done"] = "idle"
cfg["eval"] = {"every_hours": 1, "games": 200, "opponents": ["simple"], "foulplay": {"games": 0, "every_hours": 99, "parallel": 8}}
cfg["_notes"] = {
    "what this is": "Arm A of the imitation test (2026-10-14): imitation of PSBot from random weights on release baseline-c988471, "
                    "25.4M decisions (the old run's imitation length), then stop. Arm B is the same network trained on top-player "
                    "games. Each is judged by 800 games against Foul Play at 200 ms; B must beat A by 8 points at p < 0.01 to be "
                    "the start of the full run, else the full run starts from A.",
}
json.dump(cfg, open("/home/ubuntu/runs/armA/run.json", "w"), indent=1)
print("run.json written")
EOF
sudo tee /etc/systemd/system/psppo-arma-train.service >/dev/null <<EOF
[Unit]
Description=ps-ppo imitation test arm A: training supervisor (pipeline.py)
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=1800
StartLimitBurst=10
[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=$REL
Environment=PYTHONUTF8=1 PYTHONUNBUFFERED=1
EnvironmentFile=-/home/ubuntu/psppo-base.env
ExecCondition=/home/ubuntu/venv/bin/python $REL/pipeline.py --config $RUN/run.json --run-dir $RUN --can-start
ExecStart=/home/ubuntu/venv/bin/python $REL/pipeline.py --config $RUN/run.json --run-dir $RUN
Restart=always
RestartSec=30
RestartPreventExitStatus=2 3 4 5
KillSignal=SIGTERM
KillMode=mixed
TimeoutStopSec=600
LimitNOFILE=1048576
TasksMax=infinity
OOMPolicy=continue
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
/home/ubuntu/venv/bin/python $REL/pipeline.py --config $RUN/run.json --run-dir $RUN --can-start && echo "can start" || echo "can-start said no ($?)"
