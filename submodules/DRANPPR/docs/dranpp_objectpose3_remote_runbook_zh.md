# DRAN++ ObjectPose3 远端运行手册（10 机 / 15 物体）

## 1. 目的与适用范围

本文供远端执行端通过 SSH 检查实物数据、构建程序、运行当前
SE(3)-ObjectPose3 几何精度最优的 DRAN++ 配置，并回传可审计结果。

本文的默认实验规模为：

- 机器人：10 台，编号 `0..9`
- 静态物体：15 个
- 状态模型：机器人 `Pose3`，物体 `ObjectPose3`
- 物体关联：外部给定、运行期间固定
- 通信拓扑：Complete
- 通信策略：Full communication
- 迭代轮数：20
- 不增加随机扰动，不使用鲁棒核，不运行 upper association

默认算法是 **DRAN++ ObjectPose3 A18/DPCG1 精度候选版**：

| 参数 | 值 |
| --- | --- |
| `beta` | `0.075` |
| physical dual step | `0.75` |
| proximal curvature scale | `0.75` |
| local solver | `reduced_rotation` |
| local iterations | `3` |
| ObjectPose3 interface | `matrix_free_objectpose3_pcg` |
| PCG iterations | `1` |
| PCG preconditioner | `private_block_jacobi` |
| private solver | `block_tridiagonal_ldlt` |
| response period | `5` |
| terminal local polish | `1`，替代最后一轮 |

该配置在现有 D1/D2/D3 验证中同时改善了机器人和物体的平移、旋转
RMSE，且没有改变对应冻结版本的逻辑通信量。它不是所有指标上的严格
支配解：冻结版 `beta=0.1, dual_step=1.25` 在 D1、D3 的 factor cost
略低。因此，本手册把 `0.075/0.75` 作为精度优先默认值，并在第 10 节
保留冻结参数作为对照。

重要限制：以上参数在 MR.CLAM-derived Synthetic SE(3)-ObjectPose3 上
验证过，但尚未在这套 10 机实物数据上验证。远端第一次运行不得根据
真值调参。

## 2. 远端执行端必须先确认的事项

在运行前逐项报告，不能根据文件名猜测：

1. 实际文件是 `rob_0.g2o`，还是类似 `robo_0.gwo` 的名称。
2. 十个文件是否连续覆盖机器人 `0..9`，且都不是空文件。
3. 数据是否真的是 SE(3)，而不是将 SE(2) 数据补零得到的伪 SE(3)。
4. 物体观测是否提供完整 6-DoF 相对位姿。
5. 15 个物体的坐标系定义是否跨机器人一致。
6. 是否存在物体旋转对称性、标签正反面歧义或四元数跳变。
7. 每台机器人是否都包含 15 个物体后缀顶点。
8. 每台机器人实际观测了哪些物体。
9. 不同机器人的局部物体槽位是否采用相同顺序。
10. 是否已有可信的跨机器人固定关联文件。
11. g2o 信息矩阵是 information matrix，而不是 covariance matrix。
12. 是否有轨迹和物体真值；若有，真值坐标系和时间同步方式是什么。

如果物体只有三维位置，没有可靠方向，则停止使用本文配置。不能用单位
旋转伪造 ObjectPose3 观测，应改用 Point3/Point2 模型或扩展对应因子。

## 3. 源码和二进制版本

算法没有单独的可执行文件，集成在：

```text
build/bin/manual-object-dran-example
```

关键源码：

```text
include/DPGO/ManualObjectDran.h
src/ManualObjectDran.cpp
examples/ManualObjectDranExample.cpp
src/ObjectPGOData.cpp
benchmarks/mrclam_se3/run.py
scripts/run_mrclam_se3_dran_curvature_gate.py
```

当前最优实现可能包含尚未提交到公共分支的工作树修改。远端不能只根据
一个旧 commit 重新 clone 后直接声明“相同算法”。推荐同步本机工作树的
上述源码并在远端重新构建，或者同步经过确认的完整源码快照。

远端必须记录：

```bash
git rev-parse HEAD
git status --short
sha256sum \
  include/DPGO/ManualObjectDran.h \
  src/ManualObjectDran.cpp \
  examples/ManualObjectDranExample.cpp \
  src/ObjectPGOData.cpp
```

构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target manual-object-dran-example -j2
test -x build/bin/manual-object-dran-example
sha256sum build/bin/manual-object-dran-example
```

确认源码包含所需模式：

```bash
rg -n \
  "matrix_free_objectpose3_pcg|block_tridiagonal_ldlt|private_block_jacobi" \
  include/DPGO src examples
```

## 4. 数据目录接口

### 4.1 文件名

求解器只识别以下精确名称：

```text
<DATA_DIR>/
  rob_0.g2o
  rob_1.g2o
  ...
  rob_9.g2o
```

`robo_0.gwo`、`robot_0.g2o`、`rob0.g2o` 都不会被识别。

不要直接修改原始数据。若原始文件确实名为 `robo_0.gwo`，应建立 staging
目录并使用软链接：

```bash
RAW_DIR=/absolute/path/to/raw_data
DATA_DIR=/absolute/path/to/dran_input
mkdir -p "$DATA_DIR"

# 只有确认十个原文件的真实命名后才执行。
for i in $(seq 0 9); do
  ln -s "$RAW_DIR/robo_${i}.gwo" "$DATA_DIR/rob_${i}.g2o"
done
```

如果原文件已经是 `rob_0.g2o` 至 `rob_9.g2o`，直接将 `DATA_DIR` 指向
该目录，不需要重命名。

检查：

```bash
find "$DATA_DIR" -maxdepth 1 -type f -o -type l
for i in $(seq 0 9); do
  test -s "$DATA_DIR/rob_${i}.g2o" ||
    { echo "missing or empty rob_${i}.g2o"; exit 1; }
done
```

### 4.2 每个 g2o 文件的状态排列

每个 `rob_i.g2o` 同时包含该机器人的轨迹状态和 15 个局部物体副本。
求解器按下面的规则解释顶点：

```text
object_start = max_vertex_id + 1 - 15
trajectory vertices: vertex_id < object_start
object slot k:        vertex_id = object_start + k, k=0..14
```

因此必须满足：

- 最后 15 个顶点 ID 连续存在。
- 这 15 个后缀顶点只能表示物体，不能混入轨迹节点。
- 轨迹顶点 ID 必须小于 `object_start`。
- `object_start` 可以因机器人轨迹长度不同而不同。
- 关联 CSV 中的 `receiver_object` 和 `sender_object` 是槽位 `0..14`，
  不是 g2o 顶点 ID。
- 即使某台机器人没有观测某个物体，也应保留对应的物体后缀顶点；
  但不能为不存在的观测伪造 pose-object 因子。

### 4.3 SE(3) 顶点格式

轨迹和物体都使用标准 g2o `VERTEX_SE3:QUAT`：

```text
VERTEX_SE3:QUAT id x y z qx qy qz qw
```

四元数顺序必须是 `qx qy qz qw`，且范数应接近 1。初值必须有限。
每个机器人文件内的轨迹初值和物体初值应处于同一个机器人局部坐标系。

### 4.4 SE(3) 边格式

轨迹里程计/闭环边和 pose-object 观测都使用：

```text
EDGE_SE3:QUAT i j dx dy dz dqx dqy dqz dqw \
  I11 I12 I13 I14 I15 I16 \
  I22 I23 I24 I25 I26 \
  I33 I34 I35 I36 \
  I44 I45 I46 I55 I56 I66
```

每条边共 31 个字段。测量采用标准 g2o 约定，表示从端点 `i` 到端点
`j` 的相对变换。

求解器按端点类型分类：

- `trajectory -> trajectory`：该机器人内部的轨迹因子。
- `trajectory -> object`：完整 ObjectPose3 观测。
- `object -> trajectory`：允许，读取时自动求逆。
- `object -> object`：当前读取器忽略。

不要把跨机器人轨迹边直接写入某个 `rob_i.g2o`。当前目录读取器会把
该文件中的两个轨迹端点都归为机器人 `i`。若数据包含真实的跨机器人
pose-pose 因子，需要先明确扩展输入接口，不能静默丢失或错误归属。

### 4.5 信息矩阵的实际使用方式

输入必须提供 information matrix。当前读取器将 6x6 信息矩阵压缩为
DPGO 的两个标量权重：

```text
tau   <- translation information block I(0:2, 0:2)
kappa <- rotation information block I(3:5, 3:5)
```

平移-旋转交叉块会被解析但不会进入这两个标量权重。若远端数据依赖很强
的平移-旋转相关项，应在报告中明确指出，因为这会造成与原始全信息因子
不完全等价。

严禁将 covariance matrix 直接填入上述 21 个字段。若原始系统输出协方差，
应先求逆得到 information matrix，并检查正定性。

## 5. 固定物体关联接口

默认算法不估计 association。`fixed_associations.csv` 是受信任输入：

```text
receiver_robot,sender_robot,receiver_object,sender_object,weight
0,1,0,4,1
1,0,4,0,1
...
```

约束如下：

- 机器人编号范围：`0..9`
- 物体槽位范围：`0..14`
- `weight` 必须是有限非负数；确定关联使用 `1`
- 行是有方向的；同一个物理关联通常必须同时写两个方向
- 不要写 `receiver_robot == sender_robot`
- 不要默认不同机器人使用相同局部槽位顺序
- 同一 `(receiver_robot, sender_robot, receiver_object)` 不应对应多个
  权重为 1 的 sender 槽位
- 第六列 `true_object_id` 可选，求解器只读取前五列；实物实验建议省略
  真值列，避免把评估信息误当作算法输入

如果十台机器人都观测全部 15 个物体，且已确认所有局部槽位完全一致，
Complete 有向关联应有：

```text
10 * 9 * 15 = 1350 positive rows
```

只有在完成上述确认后，才能生成 identity-slot 关联：

```bash
ASSOC_FILE=/absolute/path/to/fixed_associations.csv
python - "$ASSOC_FILE" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, "w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow([
        "receiver_robot",
        "sender_robot",
        "receiver_object",
        "sender_object",
        "weight",
    ])
    for receiver in range(10):
        for sender in range(10):
            if receiver == sender:
                continue
            for object_id in range(15):
                writer.writerow(
                    [receiver, sender, object_id, object_id, 1]
                )
PY
```

如果只有部分机器人观测某物体，应只为真实、可信、双端存在的关联写正
权重。每个物理物体在观测它的机器人子图上应当连通。对于 Complete
拓扑，推荐为每对观测该物体的机器人写双向关联。

## 6. SSH 预检查

建议在 `tmux` 中操作，避免 SSH 中断终止实验：

```bash
ssh <user>@<host>
tmux new -s dran-objectpose3
```

设置路径：

```bash
export DPGO_ROOT=/absolute/path/to/dpgo_ICRA
export DATA_DIR=/absolute/path/to/dran_input
export ASSOC_FILE=/absolute/path/to/fixed_associations.csv
export OUTPUT_DIR=/absolute/path/to/results/dranpp_objectpose3_b075_d075_run1
cd "$DPGO_ROOT"
```

文件类型和 token 审计：

```bash
for i in $(seq 0 9); do
  f="$DATA_DIR/rob_${i}.g2o"
  echo "===== $f ====="
  awk '
    $1=="VERTEX_SE3:QUAT" {v++}
    $1=="EDGE_SE3:QUAT" {e++}
    $1=="VERTEX_SE2" || $1=="EDGE_SE2" {se2++}
    END {
      printf("vertices=%d edges=%d se2_records=%d\n", v, e, se2)
      if (v == 0 || e == 0 || se2 != 0) exit 1
    }
  ' "$f" || exit 1
done
```

检查字段数：

```bash
for i in $(seq 0 9); do
  f="$DATA_DIR/rob_${i}.g2o"
  awk '
    $1=="VERTEX_SE3:QUAT" && NF!=9 {
      print FILENAME ":" FNR ": bad vertex field count " NF
      bad=1
    }
    $1=="EDGE_SE3:QUAT" && NF!=31 {
      print FILENAME ":" FNR ": bad edge field count " NF
      bad=1
    }
    END {exit bad}
  ' "$f" || exit 1
done
```

逐机器人报告最后 15 个顶点和因子分类：

```bash
for i in $(seq 0 9); do
  f="$DATA_DIR/rob_${i}.g2o"
  max_id=$(awk '$1=="VERTEX_SE3:QUAT" && $2>m {m=$2} END{print m}' "$f")
  object_start=$((max_id + 1 - 15))
  awk -v start="$object_start" '
    $1=="VERTEX_SE3:QUAT" {
      vertices[$2]=1
      if ($2 < start) trajectory_vertices++
      else if ($2 < start+15) object_vertices++
    }
    $1=="EDGE_SE3:QUAT" {
      a=($2>=start && $2<start+15)
      b=($3>=start && $3<start+15)
      if (!a && !b) trajectory_edges++
      else if (a != b) object_observations++
      else object_object_edges++
    }
    END {
      for (k=0; k<15; ++k)
        if (!vertices[start+k]) missing++
      printf(
        "trajectory_vertices=%d object_vertices=%d "
        "trajectory_edges=%d object_observations=%d "
        "object_object_edges=%d missing_suffix=%d\n",
        trajectory_vertices, object_vertices, trajectory_edges,
        object_observations, object_object_edges, missing
      )
      if (missing || trajectory_edges==0 || object_observations==0) exit 1
    }
  ' "$f" || exit 1
done
```

关联文件基础检查：

```bash
head -5 "$ASSOC_FILE"
awk -F, '
  NR==1 {next}
  NF<5 {print "bad column count at line", NR; bad=1; next}
  $1<0 || $1>=10 || $2<0 || $2>=10 {
    print "bad robot index at line", NR; bad=1
  }
  $3<0 || $3>=15 || $4<0 || $4>=15 {
    print "bad object slot at line", NR; bad=1
  }
  $1==$2 {print "self association at line", NR; bad=1}
  $5<0 {print "negative weight at line", NR; bad=1}
  END {print "association_rows=" NR-1; exit bad}
' "$ASSOC_FILE"
```

上述检查通过仍不代表 association 正确。远端必须另外报告局部槽位到
物理物体 ID 的映射来源，例如 AprilTag ID、人工标注或已审计的数据关联
结果。

## 7. 默认运行命令

输出目录必须是新的目录。不要覆盖旧结果：

```bash
test ! -e "$OUTPUT_DIR" ||
  { echo "OUTPUT_DIR already exists: $OUTPUT_DIR"; exit 1; }
mkdir -p "$(dirname "$OUTPUT_DIR")"
```

运行当前精度优先配置：

```bash
build/bin/manual-object-dran-example \
  --data_dir "$DATA_DIR" \
  --num_robots 10 \
  --num_objects 15 \
  --iters 20 \
  --quiet false \
  --topology complete \
  --communication_policy all \
  --object_initialization input_state \
  --object_initialization_anchor_mode constant \
  --object_orientation_observed true \
  --object_communication_representation lifted_pose \
  --object_interface_mode matrix_free_objectpose3_pcg \
  --local_solver reduced_rotation \
  --local_max_iterations 3 \
  --local_initial_max_iterations 0 \
  --local_initial_rounds 0 \
  --reduced_rotation_preconditioner cholesky \
  --association_policy manual \
  --association_mode fixed_file_soft \
  --association_file "$ASSOC_FILE" \
  --association_freeze_upper_variables true \
  --association_lower_shared_latent_mode direct_edge_dual \
  --beta 0.075 \
  --association_lower_direct_edge_synchronous_shared_only true \
  --association_lower_direct_edge_exact_proximal_curvature false \
  --association_lower_direct_edge_proximal_curvature_scale 0.75 \
  --association_lower_direct_edge_dual_step_size 0.75 \
  --association_lower_direct_edge_dual_relaxation_residual_threshold 0 \
  --association_lower_direct_edge_spectral_dual_step false \
  --association_lower_direct_edge_global_spectral_dual_step false \
  --association_lower_direct_edge_adaptive_beta false \
  --association_require_observed_objects true \
  --association_residual_use_transform_alignment true \
  --association_transform_dimension 3 \
  --association_gauge_synchronization_enabled true \
  --association_gauge_synchronization_apply_to_states true \
  --association_gauge_synchronization_apply_once_to_initial_states false \
  --object_interface_response_period 5 \
  --object_interface_pcg_iterations 1 \
  --object_interface_pcg_preconditioner private_block_jacobi \
  --object_interface_pcg_private_solver block_tridiagonal_ldlt \
  --object_interface_pcg_post_activation_local_iterations 1 \
  --object_interface_pcg_terminal_local_polish_iterations 1 \
  --object_interface_pcg_replace_final_round_with_terminal_polish true \
  --object_interface_pcg_min_translation_spread 0.06 \
  --object_interface_pcg_min_rotation_spread_deg 1.0 \
  --object_interface_pcg_min_initial_relative_decrease 0.01 \
  --object_interface_pcg_min_continuation_relative_decrease 0.0005 \
  --object_interface_step_gain 1.0 \
  --object_pose_tol 0 \
  --object_max_age 0 \
  --diagnostic_profile lightweight \
  --association_save_iteration_estimate_snapshots false \
  --output_dir "$OUTPUT_DIR" \
  --save true \
  2>&1 | tee "${OUTPUT_DIR}.console.log"
```

必须紧接着检查管道中求解器的退出码。bash 使用：

```bash
solver_status=${PIPESTATUS[0]}
test "$solver_status" -eq 0 || {
  echo "solver failed with status $solver_status"
  exit "$solver_status"
}
```

zsh 使用：

```bash
solver_status=$pipestatus[1]
test "$solver_status" -eq 0 || {
  echo "solver failed with status $solver_status"
  exit "$solver_status"
}
```

也可以在运行前执行 `set -o pipefail`，让管道返回求解器失败状态。

该命令固定执行 20 轮，不使用基于真值或全局 cost 的提前停止。终端
polish 会替代最后一轮常规 PCG/interface 更新。

通信解释：

- direct-edge lower backend 只发送共享物体端点，不发送 target、
  certificate 或 edge-dual。
- `matrix_free_objectpose3_pcg` 每五轮会产生额外的 ObjectPose3 interface
  PCG 消息；这些消息必须计入通信量，不能只统计普通物体端点。

## 8. 输出和验收

成功运行至少应产生：

```text
<OUTPUT_DIR>/
  iteration_summary.csv
  selection_summary.csv
  object_interface_summary.csv
  object_interface_response_summary.csv
  final_association_weights.csv
  final_association_transforms.csv
  final_robot_gauges.csv
  object_poses.txt
```

`object_poses.txt` 格式：

```text
type robot_id local_vertex_id object_id x y z qx qy qz qw
trajectory ...
object ...
```

它保留每台机器人的轨迹和 15 个物体副本。默认命令将 gauge
synchronization 应用到状态，但仍应使用 `final_robot_gauges.csv` 审计
坐标变换是否正常。

基础验收：

```bash
test -s "$OUTPUT_DIR/iteration_summary.csv"
test -s "$OUTPUT_DIR/object_poses.txt"
test -s "$OUTPUT_DIR/object_interface_response_summary.csv"

grep -Eini 'nan|inf|error|exception|failed' \
  "${OUTPUT_DIR}.console.log" "$OUTPUT_DIR"/*.csv || true

tail -2 "$OUTPUT_DIR/iteration_summary.csv"
wc -l "$OUTPUT_DIR/object_poses.txt"
sha256sum "$OUTPUT_DIR"/* "${OUTPUT_DIR}.console.log"
```

期望的 `object_poses.txt` 行数为：

```text
1 header
+ all trajectory rows from 10 robots
+ 10 * 15 object-copy rows
```

需要从 `iteration_summary.csv` 提取：

- 初始和最终 `measurement_cost`
- 初始和最终 `global_cost`
- 每轮及累计 solver time
- 每轮及累计通信量
- active consensus pair 数
- observed/relay/known object copy 数

需要从 `object_interface_response_summary.csv` 提取：

- interface 是否在预期轮次激活
- PCG iteration 数
- interface 通信字节和消息数
- terminal polish 是否在第 20 轮替代常规更新
- `block_tridiagonal_ldlt` 是否出现 fallback

若存在真值：

- 轨迹 RMSE 必须先做统一的 SE(3) gauge 对齐，不能对每台机器人独立对齐。
- 不允许使用带尺度的 Sim(3) 对齐。
- 物体平移 RMSE 应在统一坐标系下，对同一物体的有效机器人副本进行平均
  后计算。
- ObjectPose3 旋转平均必须在 SO(3) 上进行，不能直接逐元素平均四元数。
- association 已固定时，评估不能再对全部物体做无约束 Hungarian 重排。
- 若 association 文件明确允许少量已审计置换，评估只能应用这些置换。

没有真值时，cost 下降和副本一致性只能说明内部优化收敛，不能替代
轨迹 RMSE 或物体 RMSE。

## 9. 必须停止并报告的异常

出现以下任一情况，不要继续调参掩盖问题：

- 文件名或扩展名与读取器约定不一致。
- 某个机器人文件缺少 15 个连续物体后缀顶点。
- 物体只有位置，没有完整相对方向。
- 物体方向因对称性在不同机器人间不一致。
- association 只提供单向行，或局部槽位映射未经验证。
- g2o 中填入 covariance 而非 information。
- 信息矩阵非有限、非正定或量级明显异常。
- 某台机器人没有 trajectory-trajectory 因子。
- 某台机器人没有任何 pose-object 因子。
- 把跨机器人 pose-pose 边错误写进单个机器人文件。
- 初始 cost 已经是 NaN/Inf。
- cost 突增多个数量级。
- PCG/private solver 大量 fallback。
- 结果只有 cost，没有保存 `object_poses.txt`。

## 10. 冻结参数对照

若默认精度版运行正常，建议在新的输出目录额外运行一次冻结参数对照。
命令保持完全相同，只修改：

```text
--beta 0.1
--association_lower_direct_edge_dual_step_size 1.25
```

两次运行必须使用相同的输入文件、association、初值、迭代数和机器环境。
比较时报告 cost、机器人 RMSE、物体平移/旋转 RMSE、time 和通信量，
不要只根据单一指标选择结果。

## 11. 远端回传清单

远端执行端最终应回传：

```text
1. hostname、CPU、内存、操作系统
2. git commit、git status
3. 四个关键源码文件和二进制的 SHA-256
4. 十个输入 g2o 的实际文件名、SHA-256、字节数
5. 每台机器人：
   trajectory vertices / object vertices /
   trajectory edges / object observation edges
6. object_start 和 15 个局部槽位到物理物体 ID 的映射
7. association 文件、行数、SHA-256、关联来源
8. 完整运行命令和退出码
9. console log
10. 完整 OUTPUT_DIR
11. 初始/最终 cost、time、通信量
12. 有真值时的统一评估结果和评估命令
13. 所有 warning、fallback 和未满足的检查项
```

建议打包：

```bash
tar -czf "${OUTPUT_DIR}.tar.gz" \
  "${OUTPUT_DIR}.console.log" \
  "$OUTPUT_DIR" \
  "$ASSOC_FILE"
sha256sum "${OUTPUT_DIR}.tar.gz"
```

不得只回传截图或手工摘录的最终数字。
