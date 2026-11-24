# Modules

This directory contains source modules shipped with SEED-SLAM.

`DBA-DRANPP/` combines the final DBA network-partition upper level with the
latest object-enabled DRAN++ distributed lower-level optimizer. It is vendored
so the SEED-SLAM repository remains portable even before DBA-DRANPP has a
public remote of its own.

The exact source revision is recorded in `DBA-DRANPP.version`.

`DRANPPR/` is a separate experimental robust/online/asynchronous research
snapshot, not a replacement for the package above. Start with
[`DRANPPR/RELEASE.md`](DRANPPR/RELEASE.md); its content provenance is recorded
in `DRANPPR/SNAPSHOT.json` and `DRANPPR/SOURCE_SHA256SUMS`.
