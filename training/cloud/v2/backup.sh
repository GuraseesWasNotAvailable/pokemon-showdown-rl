#!/bin/bash
# Copies every run's files (checkpoints, logs, evaluation results) to S3 every 10 minutes, never deleting
# anything there (no --delete; the bucket is versioned and the box's role cannot delete). Unit psppo-v2-backup.
B=${PSPPO_BUCKET:?set PSPPO_BUCKET to your S3 bucket name}
HOST=${BACKUP_NAME:-$(hostname)}
while true; do
  aws s3 sync ~/runs/v2 s3://$B/$HOST/runs/v2 --only-show-errors --exclude "*/.inductor*" --exclude "*.tmp" \
    && echo "$(date -u +%FT%TZ) synced" || echo "$(date -u +%FT%TZ) sync failed"
  sleep 600
done
