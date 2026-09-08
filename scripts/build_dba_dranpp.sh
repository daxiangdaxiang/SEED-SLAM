#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/submodules/DBA-DRANPP"
build_dir="${repo_root}/build/DBA-DRANPP"

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_DPGO_TESTS=ON
cmake --build "${build_dir}" \
  --target manual-object-dran-example testDPGO -j"${BUILD_JOBS:-2}"
