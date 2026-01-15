#!/usr/bin/env bash
set -euo pipefail

# Project 1544 (MSPC001544) Jan2026 results import helper
#
# This script is intentionally "just paths + CLI calls" so we can keep a record
# of what was imported for a given project while the Python import routines
# evolve inside the `ispec` package.
#
# Results dir (realpath):
#   /home/alex/amms06/mnt/e/MSPC001544/results/MSPC1544/Jan2026
#
# Notes:
# - Only "user-facing" artifacts are attached to project files:
#   PNG/PDF/TSV/TAB (default), excluding cache files like SQLITE/RDS.
# - Volcano-style TSVs (with a GeneID column) are also imported into the omics DB.
#
# Usage (recommended: write to a DB copy first):
#   cp iSPEC/data/ispec-import.db iSPEC/data/ispec-import-1544.db
#   iSPEC/scripts/import_project1544_mspc001544_jan2026.sh --database iSPEC/data/ispec-import-1544.db --added-by alex
#
# To preview without writing:
#   iSPEC/scripts/import_project1544_mspc001544_jan2026.sh --database iSPEC/data/ispec-import-1544.db --dry-run

PROJECT_ID="${PROJECT_ID:-1544}"

RESULTS_DIR="${RESULTS_DIR:-/home/alex/amms06/mnt/e/MSPC001544/results/MSPC1544/Jan2026}"
PREFIX="${PREFIX:-Jan2026}"

# GCT exports often live outside the date-stamped results directory.
RESULTS_GCT_DIR="${RESULTS_GCT_DIR:-/home/alex/amms06/mnt/e/MSPC001544/results/MSPC1544/export/data_gct}"
PREFIX_GCT="${PREFIX_GCT:-${PREFIX}__export__data_gct}"

# Keep cache files out of the DB by default.
INCLUDE_EXTS="${INCLUDE_EXTS:-png,pdf,tsv,tab,gct}"
EXCLUDE_EXTS="${EXCLUDE_EXTS:-sqlite,rds}"
INCLUDE_GCT_EXTS="${INCLUDE_GCT_EXTS:-gct}"

DATABASE=""
OMICS_DATABASE=""
DRY_RUN=0
FORCE=0
ADDED_BY="${ADDED_BY:-}"
IMPORT_VOLCANO=1

usage() {
  cat <<EOF
Usage: $(basename "$0") --database <db_path> [options]

Options:
  --database <path>        SQLite DB path/URI to write to (required)
  --omics-database <path>  SQLite omics DB path/URI (defaults to ISPEC_OMICS_DB_PATH/derived)
  --prefix <name>          Prefix for stored filenames in project_file (default: ${PREFIX})
  --prefix-gct <name>      Prefix for GCT export attachments (default: ${PREFIX_GCT})
  --added-by <username>    Record in prjfile_AddedBy for attachments
  --no-volcano             Do not import volcano TSVs into the omics DB
  --dry-run                Do not write; report what would be imported
  --force                  Overwrite attachments that share the same stored name

Environment overrides:
  PROJECT_ID, RESULTS_DIR, PREFIX,
  RESULTS_GCT_DIR, PREFIX_GCT,
  INCLUDE_EXTS, INCLUDE_GCT_EXTS, EXCLUDE_EXTS,
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
    --prefix)
      PREFIX="${2:-}"
      shift 2
      ;;
    --prefix-gct)
      PREFIX_GCT="${2:-}"
      shift 2
      ;;
    --added-by)
      ADDED_BY="${2:-}"
      shift 2
      ;;
    --no-volcano)
      IMPORT_VOLCANO=0
      shift
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
if [[ "$IMPORT_VOLCANO" -eq 0 ]]; then
  COMMON_ARGS+=(--no-import-volcano)
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
echo "Results dir:   ${RESULTS_DIR}"
echo "Prefix:        ${PREFIX}"
echo "GCT dir:       ${RESULTS_GCT_DIR}"
echo "GCT prefix:    ${PREFIX_GCT}"
echo "Include exts:  ${INCLUDE_EXTS}"
echo "GCT exts:      ${INCLUDE_GCT_EXTS}"
echo "Exclude exts:  ${EXCLUDE_EXTS}"
if [[ -n "$ADDED_BY" ]]; then
  echo "Added by:      ${ADDED_BY}"
fi
echo "Import volcano: ${IMPORT_VOLCANO}"
echo "Dry run:       ${DRY_RUN}"
echo "Force:         ${FORCE}"
echo

run_cmd "$ISPEC_BIN" db import-results \
  --project-id "$PROJECT_ID" \
  --results-dir "$RESULTS_DIR" \
  --prefix "$PREFIX" \
  "${COMMON_ARGS[@]}"

if [[ -d "$RESULTS_GCT_DIR" ]]; then
  echo
  echo "== Import: GCT exports =="

  GCT_ARGS=(
    --database "$DATABASE"
    --include-ext "$INCLUDE_GCT_EXTS"
    --exclude-ext "$EXCLUDE_EXTS"
    --no-import-volcano
  )
  if [[ -n "$OMICS_DATABASE" ]]; then
    GCT_ARGS+=(--omics-database "$OMICS_DATABASE")
  fi
  if [[ -n "$ADDED_BY" ]]; then
    GCT_ARGS+=(--added-by "$ADDED_BY")
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    GCT_ARGS+=(--dry-run)
  fi
  if [[ "$FORCE" -eq 1 ]]; then
    GCT_ARGS+=(--force)
  fi

  run_cmd "$ISPEC_BIN" db import-results \
    --project-id "$PROJECT_ID" \
    --results-dir "$RESULTS_GCT_DIR" \
    --prefix "$PREFIX_GCT" \
    "${GCT_ARGS[@]}"
else
  echo
  echo "Skipping GCT import (dir not found): ${RESULTS_GCT_DIR}" >&2
fi
