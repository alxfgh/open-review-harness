#!/usr/bin/env bash
# Export the Retraction Watch retractions already synced by services/api into the local cache, so
# `python -m review_harness.rw_arxiv` can read them without downloading the dataset again.
#
# The output goes to .cache/, which is not committed: Retraction Watch's reuse terms are still
# unconfirmed (see the root README), so only derived counts and identifiers leave this machine.
#
# Usage, from harness/:  bash scripts/export_rw_retractions.sh [container] [output]
set -euo pipefail

CONTAINER="${1:-auto_peer_review-db-1}"
OUT="${2:-.cache/rw/rw_retractions.csv}"
mkdir -p "$(dirname "$OUT")"

docker exec "$CONTAINER" psql -U apr -d apr -v ON_ERROR_STOP=1 -q -c "\copy (
  select record_id, nature, array_to_string(reasons, ';') as reasons, title, journal, publisher,
         retraction_date, retraction_doi, original_date, original_doi
  from rw_records
  where nature = 'Retraction'
  order by record_id
) to stdout with (format csv, header true)" > "$OUT.tmp"

docker exec "$CONTAINER" psql -U apr -d apr -At -c \
  "select version || ' ' || generated_at from dataset_state where name = 'retraction_watch'" \
  > "$OUT.version"
mv "$OUT.tmp" "$OUT"
echo "Wrote $(($(wc -l < "$OUT") - 1)) retractions to $OUT (dataset $(cat "$OUT.version"))"
