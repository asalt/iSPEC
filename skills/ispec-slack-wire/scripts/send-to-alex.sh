#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="/home/alex/tools/ispec-full"
DERIVED_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
if [[ -n "${ISPEC_FULL_ROOT:-}" ]]; then
  ROOT_DIR="$(cd "${ISPEC_FULL_ROOT}" && pwd)"
elif [[ -d "${DEFAULT_ROOT}/iSPEC" ]]; then
  ROOT_DIR="${DEFAULT_ROOT}"
else
  ROOT_DIR="${DERIVED_ROOT}"
fi
ISPEC_BIN="${ROOT_DIR}/iSPEC/.venv/bin/ispec"

TO="alex"
TEXT=""
FILE_PATH=""
TITLE=""
DRY_RUN=0
TIMEOUT_SECONDS=""

usage() {
  cat <<'USAGE'
Usage:
  send-to-alex.sh --text "message"
  send-to-alex.sh --file report.pdf --text "Report is ready"

Options:
  --to ALIAS              Recipient alias (default: alex)
  --text TEXT             Slack message text or upload comment
  --file PATH             Optional file to upload
  --title TITLE           Optional file title for upload
  --dry-run               Resolve recipient/file without calling Slack
  --timeout-seconds N     Slack HTTP timeout
  -h, --help              Show help

This wrapper loads local iSPEC Slack env files when present and then delegates
to `ispec slack send` or `ispec slack upload`.

Default env lookup root: /home/alex/tools/ispec-full
Override root with: ISPEC_FULL_ROOT=/path/to/ispec-full
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to)
      TO="${2:-}"
      shift 2
      ;;
    --text)
      TEXT="${2:-}"
      shift 2
      ;;
    --file)
      FILE_PATH="${2:-}"
      shift 2
      ;;
    --title)
      TITLE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${ISPEC_BIN}" ]]; then
  echo "Missing iSPEC CLI: ${ISPEC_BIN}" >&2
  exit 1
fi

if [[ -z "${TO}" ]]; then
  echo "--to cannot be empty" >&2
  exit 2
fi

ENV_ARGS=()
for env_file in \
  "${ROOT_DIR}/.env.local" \
  "${ROOT_DIR}/.env.slack" \
  "${ROOT_DIR}/.env.slack.local" \
  "${ROOT_DIR}/iSPEC/.env.local" \
  "${ROOT_DIR}/iSPEC/.env.slack" \
  "${ROOT_DIR}/iSPEC/.env.slack.local"; do
  if [[ -f "${env_file}" ]]; then
    ENV_ARGS+=(--env-file "${env_file}")
  fi
done

COMMON_ARGS=(--to "${TO}")
if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  COMMON_ARGS+=(--timeout-seconds "${TIMEOUT_SECONDS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

if [[ -n "${FILE_PATH}" ]]; then
  UPLOAD_ARGS=(slack upload "${COMMON_ARGS[@]}" --file "${FILE_PATH}")
  if [[ -n "${TEXT}" ]]; then
    UPLOAD_ARGS+=(--text "${TEXT}")
  fi
  if [[ -n "${TITLE}" ]]; then
    UPLOAD_ARGS+=(--title "${TITLE}")
  fi
  exec "${ISPEC_BIN}" "${ENV_ARGS[@]}" "${UPLOAD_ARGS[@]}"
fi

if [[ -z "${TEXT}" ]]; then
  echo "--text is required when --file is not provided" >&2
  exit 2
fi

exec "${ISPEC_BIN}" "${ENV_ARGS[@]}" slack send "${COMMON_ARGS[@]}" --text "${TEXT}"
