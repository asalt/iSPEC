#!/usr/bin/env bash
set -euo pipefail

# Project 1427 (MSPC001427) by-lineage results import helper
#
# This script is intentionally "just paths + CLI calls" so we can keep a record
# of what was imported for a given project while the Python import routines
# evolve inside the `ispec` package.
#
# It imports three separate lineage analyses:
#   - by-lineage.HSC
#   - by-lineage.LMPP
#   - by-lineage.MPP
#
# Notes:
# - Only "user-facing" artifacts are attached to project files:
#   PNG/PDF/TSV/TAB (default), excluding cache files like SQLITE/RDS.
# - Volcano-style TSVs (with a GeneID column) are also imported into the omics DB.
#
# Usage (recommended: write to a DB copy first):
#   cp iSPEC/data/ispec-import.db iSPEC/data/ispec-import-1427.db
#   iSPEC/scripts/import_project1427_mspc001427_by_lineage.sh --database iSPEC/data/ispec-import-1427.db
#
# To preview without writing:
#   iSPEC/scripts/import_project1427_mspc001427_by_lineage.sh --database iSPEC/data/ispec-import-1427.db --dry-run

PROJECT_ID="${PROJECT_ID:-1427}"

RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/amms06/mnt/e/MSPC001427/results}"
RESULTS_HSC_DIR="${RESULTS_HSC_DIR:-${RESULTS_ROOT}/MSPC1427_20251208.by-lineage.HSC/Dec2025}"
RESULTS_LMPP_DIR="${RESULTS_LMPP_DIR:-${RESULTS_ROOT}/MSPC1427_20251208.by-lineage.LMPP/Dec2025}"
RESULTS_MPP_DIR="${RESULTS_MPP_DIR:-${RESULTS_ROOT}/MSPC1427_20251208.by-lineage.MPP/Dec2025}"

# Set explicit prefixes so the UI groups these as HSC/LMPP/MPP analyses.
PREFIX_HSC="${PREFIX_HSC:-HSC__Dec2025}"
PREFIX_LMPP="${PREFIX_LMPP:-LMPP__Dec2025}"
PREFIX_MPP="${PREFIX_MPP:-MPP__Dec2025}"

# Keep cache files out of the DB by default.
INCLUDE_EXTS="${INCLUDE_EXTS:-png,pdf,tsv,tab,gct}"
EXCLUDE_EXTS="${EXCLUDE_EXTS:-sqlite,rds}"

DATABASE=""
OMICS_DATABASE=""
DRY_RUN=0
FORCE=0
ADDED_BY="${ADDED_BY:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --database <db_path> [options]

Options:
  --database <path>        SQLite DB path/URI to write to (required)
  --omics-database <path>  SQLite omics DB path/URI (defaults to ISPEC_OMICS_DB_PATH/derived)
  --added-by <username>    Record in prjfile_AddedBy for attachments
  --dry-run                Do not write; report what would be imported
  --force                  Overwrite attachments that share the same stored name

Environment overrides:
  PROJECT_ID, RESULTS_ROOT,
  RESULTS_HSC_DIR, RESULTS_LMPP_DIR, RESULTS_MPP_DIR,
  PREFIX_HSC, PREFIX_LMPP, PREFIX_MPP,
  INCLUDE_EXTS, EXCLUDE_EXTS,
  ADDED_BY
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --database)
      DATABASE="${2:-}"
      shift 2
      ;;
    --omics-database)
      OMICS_DATABASE="${2:-}"
      shift 2
      ;;
    --added-by)
      ADDED_BY="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATABASE" ]]; then
  echo "--database is required" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISPEC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ISPEC_BIN="${ISPEC_BIN:-${ISPEC_ROOT}/.venv/bin/ispec}"
if [[ ! -x "$ISPEC_BIN" ]]; then
  if command -v ispec >/dev/null 2>&1; then
    ISPEC_BIN="ispec"
  else
    echo "Unable to find iSPEC CLI. Activate iSPEC/.venv or set ISPEC_BIN=/path/to/ispec." >&2
    exit 1
  fi
fi

COMMON_ARGS=(
  --database "$DATABASE"
  --include-ext "$INCLUDE_EXTS"
  --exclude-ext "$EXCLUDE_EXTS"
)
if [[ -n "$OMICS_DATABASE" ]]; then
  COMMON_ARGS+=(--omics-database "$OMICS_DATABASE")
fi
if [[ -n "$ADDED_BY" ]]; then
  COMMON_ARGS+=(--added-by "$ADDED_BY")
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  COMMON_ARGS+=(--dry-run)
fi
if [[ "$FORCE" -eq 1 ]]; then
  COMMON_ARGS+=(--force)
fi

run_cmd() {
  echo "+ $*"
  "$@"
}

echo "Project ID:    ${PROJECT_ID}"
echo "Database:      ${DATABASE}"
if [[ -n "$OMICS_DATABASE" ]]; then
  echo "Omics DB:      ${OMICS_DATABASE}"
fi
echo "Results (HSC):  ${RESULTS_HSC_DIR}"
echo "Results (LMPP): ${RESULTS_LMPP_DIR}"
echo "Results (MPP):  ${RESULTS_MPP_DIR}"
echo "Include exts:   ${INCLUDE_EXTS}"
echo "Exclude exts:   ${EXCLUDE_EXTS}"
if [[ -n "$ADDED_BY" ]]; then
  echo "Added by:      ${ADDED_BY}"
fi
echo "Dry run:       ${DRY_RUN}"
echo "Force:         ${FORCE}"
echo

echo "== Import: HSC =="
run_cmd "$ISPEC_BIN" db import-results \
  --project-id "$PROJECT_ID" \
  --results-dir "$RESULTS_HSC_DIR" \
  --prefix "$PREFIX_HSC" \
  "${COMMON_ARGS[@]}"
echo

echo "== Import: LMPP =="
run_cmd "$ISPEC_BIN" db import-results \
  --project-id "$PROJECT_ID" \
  --results-dir "$RESULTS_LMPP_DIR" \
  --prefix "$PREFIX_LMPP" \
  "${COMMON_ARGS[@]}"
echo

echo "== Import: MPP =="
run_cmd "$ISPEC_BIN" db import-results \
  --project-id "$PROJECT_ID" \
  --results-dir "$RESULTS_MPP_DIR" \
  --prefix "$PREFIX_MPP" \
  "${COMMON_ARGS[@]}"
echo
