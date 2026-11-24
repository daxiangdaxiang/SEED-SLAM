# SEED-SLAM

## Experimental DRANPPR Release

`v0.1.0-dranppr.1` adds an independent research snapshot in
[`submodules/DRANPPR`](submodules/DRANPPR/RELEASE.md). It includes persistent
local-factor research code, adaptive update/error models, and reproducible
asynchronous fixed-quadratic experiments. It does **not** replace the original
DBA-DRAN++ backend or make the new predictor a stable default.

The latest model-response candidate avoids the inspected long-delay divergences
but loses to adaptive hold at equal bytes (0 wins, 2 ties, 26 losses). See the
[release notes](releases/v0.1.0-dranppr.1.md) for evidence and limitations.

```sh
python3 -m pip install -r releases/dranppr-requirements.txt
bash scripts/test_dranppr_python.sh
bash scripts/reproduce_dranppr.sh
```

The pinned Python 3.9 research environment reproduces the latest 140-run study
from compact fixtures included in the repository. Native C++ and historical
dataset workflows have separate requirements documented in the package.

SEED-SLAM is the system-level workspace for distributed semantic estimation.
The current repository starts from the verified DBA and DRAN++ object-association
backend and leaves `data/` available for system datasets that are not committed
to Git.

## Layout

```text
SEED-SLAM/
|-- data/                       external SEED-SLAM datasets
|-- scripts/                    repository-level build and sync helpers
`-- submodules/
    |-- DBA-DRANPP/             original verified DBA + DRAN++ snapshot
    `-- DRANPPR/                opt-in robust/online/asynchronous research snapshot
```

`submodules/DBA-DRANPP` is a vendored source snapshot rather than a Git
submodule with a machine-local URL. Its provenance is recorded in
`submodules/DBA-DRANPP.version`, so a normal clone contains all code required
for the build.

## Build

Install the C++ dependencies first:

```bash
sudo apt-get install build-essential cmake-gui libsuitesparse-dev
```

Then configure and build the object-enabled DRAN++ backend from the repository
root:

```bash
./scripts/build_dba_dranpp.sh
```

The main executable is:

```text
build/DBA-DRANPP/bin/manual-object-dran-example
```

For the DBA fixed-point workflow, datasets, commands, verified results, and
scope limitations, see
[`submodules/DBA-DRANPP/README.md`](submodules/DBA-DRANPP/README.md).

## Refresh DBA and DRAN++

After updating the standalone `DBA-DRANPP` repository, refresh the vendored
snapshot and its provenance record with:

```bash
./scripts/sync_dba_dranpp.sh ../DBA-DRANPP
```
