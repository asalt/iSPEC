#!/usr/bin/env bash
set -euo pipefail

# Project 1494 (MSPC001494) prof/phos/GSEA results import helper
#
# This script is intentionally "just paths + CLI calls" so we can keep a record
# of what was imported for a given project while the Python import routines
# evolve inside the `ispec` package.
#
# Results root (realpath):
#   /home/alex/amms06/mnt/e/MSPC001494/results
#
# Notes:
# - Only "user-facing" artifacts are attached to project files:
#   PNG/PDF/TSV/TAB/GCT (default), excluding cache files like SQLITE/RDS.
# - `gsea/` is large (~7k PDFs / ~500MB); use --skip-gsea if you only want prof/phos.
#
# Usage (recommended: write to a DB copy first):
#   cp iSPEC/data/ispec-import.db iSPEC/data/ispec-import-1494.db
#   iSPEC/scripts/import_project1494_mspc001494_prof_phos_gsea.sh --database iSPEC/data/ispec-import-1494.db --added-by alex
#
# To preview without writing:
#   iSPEC/scripts/import_project1494_mspc001494_prof_phos_gsea.sh --database iSPEC/data/ispec-import-1494.db --dry-run

PROJECT_ID="${PROJECT_ID:-1494}"

RESULTS_ROOT="${RESULTS_ROOT:-/home/alex/amms06/mnt/e/MSPC001494/results}"

# Proteomics ("prof")
RESULTS_PROF_NOV2025_DIR="${RESULTS_PROF_NOV2025_DIR:-${RESULTS_ROOT}/MSPC1494_prof/Nov2025}"
PREFIX_PROF_NOV2025="${PREFIX_PROF_NOV2025:-prof__Nov2025}"

RESULTS_PROF_TOPDIFF_DIR="${RESULTS_PROF_TOPDIFF_DIR:-${RESULTS_ROOT}/MSPC1494_prof/topdiff}"
PREFIX_PROF_TOPDIFF="${PREFIX_PROF_TOPDIFF:-prof-topdiff}"

# Phospho ("phos")
RESULTS_PHOS_QC_DIR="${RESULTS_PHOS_QC_DIR:-${RESULTS_ROOT}/phos/qc}"
PREFIX_PHOS_QC="${PREFIX_PHOS_QC:-phos__metrics__qc}"

RESULTS_PHOS_DATASET_DIR="${RESULTS_PHOS_DATASET_DIR:-${RESULTS_ROOT}/phos/MSPC1494_prof_sty_79_9663_site_siteinfo_combined_nr_nr_54978x16}"

RESULTS_PHOS_PCA_DIR="${RESULTS_PHOS_PCA_DIR:-${RESULTS_PHOS_DATASET_DIR}/pca}"
PREFIX_PHOS_PCA="${PREFIX_PHOS_PCA:-phos__pca}"

RESULTS_PHOS_VOLCANO_DIR="${RESULTS_PHOS_VOLCANO_DIR:-${RESULTS_PHOS_DATASET_DIR}/limma/volcano}"
PREFIX_PHOS_VOLCANO="${PREFIX_PHOS_VOLCANO:-phos__volcano}"

RESULTS_PHOS_TABLES_DIR="${RESULTS_PHOS_TABLES_DIR:-${RESULTS_PHOS_DATASET_DIR}/limma/tables}"
PREFIX_PHOS_TABLES="${PREFIX_PHOS_TABLES:-phos__volcano}"

RESULTS_PHOS_CLUSTER_HEATMAP_DIR="${RESULTS_PHOS_CLUSTER_HEATMAP_DIR:-${RESULTS_PHOS_DATASET_DIR}/heatmap}"
PREFIX_PHOS_CLUSTER_HEATMAP="${PREFIX_PHOS_CLUSTER_HEATMAP:-phos__clustermap__heatmap}"

RESULTS_PHOS_CLUSTER_LIMMA_ALL_DIR="${RESULTS_PHOS_CLUSTER_LIMMA_ALL_DIR:-${RESULTS_PHOS_DATASET_DIR}/limma/all}"
PREFIX_PHOS_CLUSTER_LIMMA_ALL="${PREFIX_PHOS_CLUSTER_LIMMA_ALL:-phos__clustermap__limma_all}"

RESULTS_PHOS_CLUSTER_LIMMA_SUBSETS_DIR="${RESULTS_PHOS_CLUSTER_LIMMA_SUBSETS_DIR:-${RESULTS_PHOS_DATASET_DIR}/limma/subsets}"
PREFIX_PHOS_CLUSTER_LIMMA_SUBSETS="${PREFIX_PHOS_CLUSTER_LIMMA_SUBSETS:-phos__clustermap__limma_subsets}"

RESULTS_PHOS_EXPORT_DIR="${RESULTS_PHOS_EXPORT_DIR:-${RESULTS_ROOT}/phos/export}"
PREFIX_PHOS_EXPORT="${PREFIX_PHOS_EXPORT:-phos__export}"

# GSEA
RESULTS_GSEA_DIR="${RESULTS_GSEA_DIR:-${RESULTS_ROOT}/gsea}"
PREFIX_GSEA="${PREFIX_GSEA:-gsea}"

# Keep cache files out of the DB by default.
INCLUDE_EXTS="${INCLUDE_EXTS:-png,pdf,tsv,tab,gct}"
EXCLUDE_EXTS="${EXCLUDE_EXTS:-sqlite,rds}"

DATABASE=""
OMICS_DATABASE=""
DRY_RUN=0
FORCE=0
ADDED_BY="${ADDED_BY:-}"
IMPORT_VOLCANO=1
IMPORT_PROF=1
IMPORT_PHOS=1
IMPORT_GSEA=1

usage() {
  cat <<EOF
Usage: $(basename "$0") --database <db_path> [options]

Options:
  --database <path>        SQLite DB path/URI to write to (required)
  --omics-database <path>  SQLite omics DB path/URI (defaults to ISPEC_OMICS_DB_PATH/derived)
  --added-by <username>    Record in prjfile_AddedBy for attachments
  --no-volcano             Do not import volcano TSVs into the omics DB
  --skip-prof              Skip proteomics imports
  --skip-phos              Skip phospho imports
  --skip-gsea              Skip GSEA imports
  --dry-run                Do not write; report what would be imported
  --force                  Overwrite attachments that share the same stored name

Environment overrides:
  PROJECT_ID, RESULTS_ROOT,
  RESULTS_PROF_NOV2025_DIR, PREFIX_PROF_NOV2025,
  RESULTS_PROF_TOPDIFF_DIR, PREFIX_PROF_TOPDIFF,
  RESULTS_PHOS_QC_DIR, PREFIX_PHOS_QC,
  RESULTS_PHOS_DATASET_DIR,
  RESULTS_PHOS_PCA_DIR, PREFIX_PHOS_PCA,
  RESULTS_PHOS_VOLCANO_DIR, PREFIX_PHOS_VOLCANO,
  RESULTS_PHOS_TABLES_DIR, PREFIX_PHOS_TABLES,
  RESULTS_PHOS_CLUSTER_HEATMAP_DIR, PREFIX_PHOS_CLUSTER_HEATMAP,
  RESULTS_PHOS_CLUSTER_LIMMA_ALL_DIR, PREFIX_PHOS_CLUSTER_LIMMA_ALL,
  RESULTS_PHOS_CLUSTER_LIMMA_SUBSETS_DIR, PREFIX_PHOS_CLUSTER_LIMMA_SUBSETS,
  RESULTS_PHOS_EXPORT_DIR, PREFIX_PHOS_EXPORT,
  RESULTS_GSEA_DIR, PREFIX_GSEA,
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
    --no-volcano)
      IMPORT_VOLCANO=0
      shift
      ;;
    --skip-prof)
      IMPORT_PROF=0
      shift
      ;;
    --skip-phos)
      IMPORT_PHOS=0
      shift
      ;;
    --skip-gsea)
      IMPORT_GSEA=0
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

echo "Project ID:     ${PROJECT_ID}"
echo "Database:       ${DATABASE}"
if [[ -n "$OMICS_DATABASE" ]]; then
  echo "Omics DB:       ${OMICS_DATABASE}"
fi
echo "Results root:   ${RESULTS_ROOT}"
echo "Include exts:   ${INCLUDE_EXTS}"
echo "Exclude exts:   ${EXCLUDE_EXTS}"
if [[ -n "$ADDED_BY" ]]; then
  echo "Added by:       ${ADDED_BY}"
fi
echo "Import volcano: ${IMPORT_VOLCANO}"
echo "Dry run:        ${DRY_RUN}"
echo "Force:          ${FORCE}"
echo

if [[ "$IMPORT_PROF" -eq 1 ]]; then
  echo "== Import: prof (Nov2025) =="
  if [[ -d "$RESULTS_PROF_NOV2025_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PROF_NOV2025_DIR" \
      --prefix "$PREFIX_PROF_NOV2025" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PROF_NOV2025_DIR}" >&2
  fi
  echo

  echo "== Import: prof (topdiff) =="
  if [[ -d "$RESULTS_PROF_TOPDIFF_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PROF_TOPDIFF_DIR" \
      --prefix "$PREFIX_PROF_TOPDIFF" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PROF_TOPDIFF_DIR}" >&2
  fi
  echo
fi

if [[ "$IMPORT_PHOS" -eq 1 ]]; then
  echo "== Import: phos (qc) =="
  if [[ -d "$RESULTS_PHOS_QC_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PHOS_QC_DIR" \
      --prefix "$PREFIX_PHOS_QC" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PHOS_QC_DIR}" >&2
  fi
  echo

  echo "== Import: phos (pca) =="
  if [[ -d "$RESULTS_PHOS_PCA_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PHOS_PCA_DIR" \
      --prefix "$PREFIX_PHOS_PCA" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PHOS_PCA_DIR}" >&2
  fi
  echo

  echo "== Import: phos (volcano plots) =="
  if [[ -d "$RESULTS_PHOS_VOLCANO_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PHOS_VOLCANO_DIR" \
      --prefix "$PREFIX_PHOS_VOLCANO" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PHOS_VOLCANO_DIR}" >&2
  fi
  echo

  echo "== Import: phos (limma tables) =="
  if [[ -d "$RESULTS_PHOS_TABLES_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PHOS_TABLES_DIR" \
      --prefix "$PREFIX_PHOS_TABLES" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PHOS_TABLES_DIR}" >&2
  fi
  echo

  echo "== Import: phos (cluster heatmap) =="
  if [[ -d "$RESULTS_PHOS_CLUSTER_HEATMAP_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PHOS_CLUSTER_HEATMAP_DIR" \
      --prefix "$PREFIX_PHOS_CLUSTER_HEATMAP" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PHOS_CLUSTER_HEATMAP_DIR}" >&2
  fi
  echo

  echo "== Import: phos (cluster limma/all) =="
  if [[ -d "$RESULTS_PHOS_CLUSTER_LIMMA_ALL_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PHOS_CLUSTER_LIMMA_ALL_DIR" \
      --prefix "$PREFIX_PHOS_CLUSTER_LIMMA_ALL" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PHOS_CLUSTER_LIMMA_ALL_DIR}" >&2
  fi
  echo

  echo "== Import: phos (cluster limma/subsets) =="
  if [[ -d "$RESULTS_PHOS_CLUSTER_LIMMA_SUBSETS_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PHOS_CLUSTER_LIMMA_SUBSETS_DIR" \
      --prefix "$PREFIX_PHOS_CLUSTER_LIMMA_SUBSETS" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PHOS_CLUSTER_LIMMA_SUBSETS_DIR}" >&2
  fi
  echo

  echo "== Import: phos (export) =="
  if [[ -d "$RESULTS_PHOS_EXPORT_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_PHOS_EXPORT_DIR" \
      --prefix "$PREFIX_PHOS_EXPORT" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_PHOS_EXPORT_DIR}" >&2
  fi
  echo
fi

if [[ "$IMPORT_GSEA" -eq 1 ]]; then
  echo "== Import: gsea =="
  if [[ -d "$RESULTS_GSEA_DIR" ]]; then
    run_cmd "$ISPEC_BIN" db import-results \
      --project-id "$PROJECT_ID" \
      --results-dir "$RESULTS_GSEA_DIR" \
      --prefix "$PREFIX_GSEA" \
      "${COMMON_ARGS[@]}"
  else
    echo "Skipping: dir not found: ${RESULTS_GSEA_DIR}" >&2
  fi
  echo
fi
