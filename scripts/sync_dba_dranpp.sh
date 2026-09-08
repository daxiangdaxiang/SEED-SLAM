#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${1:-${repo_root}/../DBA-DRANPP}"
target_dir="${repo_root}/submodules/DBA-DRANPP"

if [[ ! -d "${source_dir}/.git" ]]; then
  printf 'Not a DBA-DRANPP Git repository: %s\n' "${source_dir}" >&2
  exit 2
fi

mkdir -p "${target_dir}"
rsync -a --delete \
  --exclude=.git/ \
  --exclude=.pytest_cache/ \
  --exclude=build/ \
  --exclude='build-*/' \
  --exclude='__pycache__/' \
  "${source_dir}/" "${target_dir}/"

commit="$(git -C "${source_dir}" rev-parse HEAD)"
branch="$(git -C "${source_dir}" branch --show-current)"
date="$(date +%F)"
printf '%s\n' \
  'source_repository=DBA-DRANPP' \
  "source_commit=${commit}" \
  "source_branch=${branch}" \
  "snapshot_date=${date}" \
  >"${repo_root}/submodules/DBA-DRANPP.version"

printf 'Synchronized DBA-DRANPP at %s\n' "${commit}"
