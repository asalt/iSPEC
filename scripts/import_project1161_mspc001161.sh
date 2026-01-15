#!/usr/bin/env bash
set -euo pipefail

# Project 1161 (MSPC001161) import helper
#
# This script is intentionally "just paths + CLI calls" so we can keep a record
# of what was imported for a given project while the Python import routines
# evolve inside the `ispec` package.
#
# Usage (recommended: write to a DB copy first):
#   cp iSPEC/data/ispec-import.db iSPEC/data/ispec-import-omics.db
#   iSPEC/scripts/import_project1161_mspc001161.sh --database iSPEC/data/ispec-import-omics.db
#
# To preview without writing:
#   iSPEC/scripts/import_project1161_mspc001161.sh --database iSPEC/data/ispec-import-omics.db --dry-run

PROJECT_ID="${PROJECT_ID:-1161}"

E2G_DIR="${E2G_DIR:-/mnt/e/MSPC001161/data/e2g}"
RESULTS_AVFONLY_DIR="${RESULTS_AVFONLY_DIR:-/mnt/e/MSPC001161/results/MSPC001161_2_new_AVFonly}"
RESULTS_NOAVF_DIR="${RESULTS_NOAVF_DIR:-/mnt/e/MSPC001161/results/MSPC001161_2_new_noAVF}"

# Newer pipeline outputs currently appear under this date-stamped folder.
RESULTS_AVFONLY_RUN_DIR="${RESULTS_AVFONLY_RUN_DIR:-${RESULTS_AVFONLY_DIR}/MSPC001161_20251201}"
RESULTS_NOAVF_RUN_DIR="${RESULTS_NOAVF_RUN_DIR:-${RESULTS_NOAVF_DIR}/MSPC001161_20251201}"

GSEA_2COMP_TTEST_DIR="${GSEA_2COMP_TTEST_DIR:-${RESULTS_AVFONLY_DIR}/gsea/2comp-ttest/gsea_tables}"

DATABASE=""
OMICS_DATABASE=""
DRY_RUN=0
FORCE=0
STORE_METADATA=1

usage() {
  cat <<EOF
Usage: $(basename "$0") --database <db_path> [options]

Options:
  --database <path>     SQLite DB path/URI to write to (required)
  --omics-database <path>  SQLite omics DB path/URI (defaults to ISPEC_OMICS_DB_PATH/derived)
  --dry-run             Print commands without executing
  --force               Clear + re-import where supported
  --no-metadata          Do not store metadata_json extras

# (would be good to describe them)
Environment overrides:
  PROJECT_ID, E2G_DIR, RESULTS_AVFONLY_DIR, RESULTS_NOAVF_DIR,
  RESULTS_AVFONLY_RUN_DIR, RESULTS_NOAVF_RUN_DIR, GSEA_2COMP_TTEST_DIR

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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-metadata)
      STORE_METADATA=0
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

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

echo "Project ID: ${PROJECT_ID}"
echo "Database:   ${DATABASE}"
if [[ -n "$OMICS_DATABASE" ]]; then
  echo "Omics DB:   ${OMICS_DATABASE}"
fi
echo "E2G dir:    ${E2G_DIR}"
echo "Results AVF:   ${RESULTS_AVFONLY_RUN_DIR}"
echo "Results noAVF: ${RESULTS_NOAVF_RUN_DIR}"
echo "GSEA dir:   ${GSEA_2COMP_TTEST_DIR}"
echo

E2G_ARGS=()
if [[ "$STORE_METADATA" -eq 1 ]]; then
  E2G_ARGS+=(--store-metadata)
fi
if [[ "$FORCE" -eq 1 ]]; then
  E2G_ARGS+=(--force)
fi
if [[ -n "$OMICS_DATABASE" ]]; then
  E2G_ARGS+=(--omics-database "$OMICS_DATABASE")
fi

VOLCANO_ARGS=()
if [[ "$STORE_METADATA" -eq 1 ]]; then
  VOLCANO_ARGS+=(--store-metadata)
fi
if [[ "$FORCE" -eq 1 ]]; then
  VOLCANO_ARGS+=(--force)
fi
if [[ -n "$OMICS_DATABASE" ]]; then
  VOLCANO_ARGS+=(--omics-database "$OMICS_DATABASE")
fi

GSEA_ARGS=()
if [[ "$STORE_METADATA" -eq 1 ]]; then
  GSEA_ARGS+=(--store-metadata)
fi
if [[ "$FORCE" -eq 1 ]]; then
  GSEA_ARGS+=(--force)
fi
if [[ -n "$OMICS_DATABASE" ]]; then
  GSEA_ARGS+=(--omics-database "$OMICS_DATABASE")
fi

echo "== Import: E2G (QUAL/QUANT) =="
if [[ -d "$E2G_DIR" ]]; then
  run_cmd "$ISPEC_BIN" db import-e2g --dir "$E2G_DIR" --database "$DATABASE" "${E2G_ARGS[@]}"
else
  echo "Skipping E2G: directory not found: $E2G_DIR" >&2
fi
echo

echo "== Import: Volcano (gene contrasts) =="
declare -a volcano_files=()
if [[ -d "$RESULTS_AVFONLY_RUN_DIR" ]]; then
  while IFS= read -r path; do volcano_files+=("$path"); done < <(
    find "$RESULTS_AVFONLY_RUN_DIR" -type f -path '*volcano/limma/*' -name '*_group_*_dir_*.tsv' | sort
  )
fi
if [[ -d "$RESULTS_NOAVF_RUN_DIR" ]]; then
  while IFS= read -r path; do volcano_files+=("$path"); done < <(
    find "$RESULTS_NOAVF_RUN_DIR" -type f -path '*volcano/limma/*' -name '*_group_*_dir_*.tsv' | sort
  )
fi

if [[ "${#volcano_files[@]}" -eq 0 ]]; then
  echo "No volcano TSVs found." >&2
else
  for path in "${volcano_files[@]}"; do
    run_cmd "$ISPEC_BIN" db import-volcano \
      --project-id "$PROJECT_ID" \
      --database "$DATABASE" \
      --file "$path" \
      "${VOLCANO_ARGS[@]}"
  done
fi
echo

echo "== Import: GSEA (2comp-ttest gsea_tables) =="
declare -a gsea_files=()
if [[ -d "$GSEA_2COMP_TTEST_DIR" ]]; then
  while IFS= read -r path; do gsea_files+=("$path"); done < <(
    find "$GSEA_2COMP_TTEST_DIR" -type f -name '*.tsv' | sort
  )
fi

if [[ "${#gsea_files[@]}" -eq 0 ]]; then
  echo "No GSEA TSVs found." >&2
else
  for path in "${gsea_files[@]}"; do
    run_cmd "$ISPEC_BIN" db import-gsea \
      --project-id "$PROJECT_ID" \
      --database "$DATABASE" \
      --file "$path" \
      "${GSEA_ARGS[@]}"
  done
fi
echo

cat <<'EOF'
== Notes ==
- This script imports TSV tables (E2G, volcano, GSEA). It does not import PNG/PDF figures yet.
- For figures, iSPEC currently supports storing bytes in `project_file` (see /api/projects/{id}/files).
  If we want CLI-backed figure import, we can add a `ispec db import-project-files` subcommand with size limits.
EOF
