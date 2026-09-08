# SEED-SLAM

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
    `-- DBA-DRANPP/             verified DBA + latest DRAN++ source snapshot
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
