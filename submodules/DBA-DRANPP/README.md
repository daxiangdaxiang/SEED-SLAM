# DBA-DRAN++

本仓库是当前 DBA 与 DRAN++ 的可复现实验快照。它把网络级物体关联上层
和分布式物体位姿图下层放在同一仓库中，不再依赖旧 `DRAN-SEEDSLAM`
仓库或历史结果目录。

## 当前范围

- DBA 主实验：SemanticKITTI 05/07，Point2 物体状态，变基数网络身份划分。
- DRAN++ 后端：同时保留 Point2 与 ObjectPose3 模式。
- 当前发布的 DBA 上层只对 Point2 闭环完成了验证；ObjectPose3 后端已经
  集成，但不能据此声称 ObjectPose3 关联上层也已完成验证。
- 最终 DBA 名称统一为 `DBA`。`DCPP` 仅是历史开发名称，本仓库不再使用。

## 算法链路

1. 给定当前关联划分，DRAN++ 求解分布式物体位姿图下层。
2. 对每条通信拓扑边移除其显式关联因子，导出局部二次模型预测的物体
   均值与协方差，避免候选关联直接生成验证自己的证据。
3. DBA 将全部机器人副本表示为一个变基数网络划分。上层联合使用留边外
   状态、完整机器人内协方差、语义类别、经验信息强度和共享坐标变换。
4. 在固定候选支持上生成 `merge`、`move`、`detach` 编辑，通过分布式
   min-consensus 选择互不冲突且严格降低同一网络目标的编辑批次。
5. 新划分触发下一次完整 DRAN++ 下层响应；划分哈希不再改变时停止。

详细建模和代码映射见 [docs/DBA_ALGORITHM.md](docs/DBA_ALGORITHM.md)。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DPGO_TESTS=ON
cmake --build build --target manual-object-dran-example testDPGO -j2
```

主要可执行程序：

```text
build/bin/manual-object-dran-example
```

## 复现实验

```bash
python scripts/run_dba_fixed_point.py \
  --dataset-manifest experiments/dba/05/dataset_manifest.json \
  --binary build/bin/manual-object-dran-example \
  --output-dir results/latest/05

python scripts/run_dba_fixed_point.py \
  --dataset-manifest experiments/dba/07/dataset_manifest.json \
  --binary build/bin/manual-object-dran-example \
  --output-dir results/latest/07
```

`joint_cavity_leave_pair` 是默认且唯一推荐的论文主线目标。数据集清单只
包含数据事实；算法参数由 runner 统一给出，避免把调参写入数据 manifest。

## 最新结果

`results/latest/` 只保留用本仓库源码重新生成的 05/07 最终闭环结果。
关联输出和旧后端完全一致，而最新版后端降低了记录的通信量。

| 数据集 | Association F1 | Network partition F1 | 外层响应数 | 旧/新总通信 MB | 最终下层 measurement cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| 05 | 0.857143 | 0.680739 | 2 | 5.578486 / 4.664699 | 208.983053 |
| 07 | 0.969543 | 0.930233 | 1 | 2.030701 / 1.809326 | 257.100478 |

05 与 07 的最终划分哈希和所有关联指标逐项复现旧版本。07 的下层 cost
相对旧后端增加 `0.037455`（约 `0.0146%`），因此这里只声称关联结果保持、
通信下降，不声称新后端在每个下层目标上严格支配旧后端。

## 验证

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_dba_*.py tests/test_run_dba_*.py \
  tests/test_analyze_dba_feedback_safe_ablation.py \
  tests/test_audit_dba_*.py

build/bin/testDPGO --gtest_filter='*TrackEmpirical*'
```

本次快照的 DBA Python 测试为 `353 passed`，恢复接口的 C++ 聚焦测试为
`7 passed`。完整 DRAN++ C++ 套件共 900 项，其中 6 项来自所选最新版
工作树的既有失败：1 项机器相关的 60 ms 性能阈值和 5 项实验性
DRAN++/SyntheticObjectPGO 断言。它们未被隐藏或通过放宽断言修改。

版本选择、源码哈希和差异见 [docs/VERSION_MIGRATION.md](docs/VERSION_MIGRATION.md)。

