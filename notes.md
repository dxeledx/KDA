# Notes: KSDA 当前研究进展综合

## 已确认的主线
- 当前不可偏离的目标不是继续堆 controller，而是：
  1. 定义并验证脑电表征—行为不一致性；
  2. 在 Koopman 特征空间做对齐；
  3. 再讨论用线性性能指标修正 Koopman 算子。
- 这条主线已经在 `docs/KSDA/00-overview-v2.md`、`docs/KSDA/22-primary-goal-mismatch-first.md`、`docs/KSDA/KSDA.md` 中反复固定。

## 已有关键证据
- Stage 2 已经说明：连续动态 gating + 行为反馈不是当前瓶颈解法，论文叙事转向 `RA-first / safe adaptation`。
- `D1-T rank=48` 成功把动作空间拉开，说明表征瓶颈真实存在，动作集本身已可用。
- `D1.5` 失败说明当前瓶颈不是动作，而是 teacher / supervisory signal 的构造。
- `RBID` 已能支撑“问题存在且现有方法仍存在该问题”。
- `K-RBID` 目前只有中等强度的局部诊断价值，暂不适合直接驱动 online update。

## 当前最自然的 paper-facing 主实验
- 先做一个保守的 `RBID-aware Koopman aligner`：
  - 静态或窗口级；
  - 目标是降低 `RBID / Tail-RBID` 并提高 `LOSO accuracy`；
  - 不把 `teacher signal`、`trial-level selector`、`online controller` 绑进主实验。

## 当前最重要的实验设计原则
- 每一轮只改一个主因素。
- 所有方法统一报告：`accuracy / kappa / RBID / Tail-RBID`。
- 如果新方法只能改善几何相似性，不能改善行为排序或 accuracy，则不能算主线进展。
- 若 `K-RBID` 仍然不稳定，则保留为诊断工具，不强行方法化。

## E0 刷新结果（2026-03-09）
- 运行目录：`results/e0/2026-03-09-e0-r1`
- 主表已经固定在：`results/e0/2026-03-09-e0-r1/summary/e0_main_table.csv`
- 当前主表中：
  - `RA` 仍是最高 LOSO accuracy：`0.4375`
  - `EA` 与 `RA` 拿到最低 `RBID`：`0.2817`
  - `Koopman-noalign-r48`：`accuracy=0.3943`, `RBID=0.3413`
  - `Static Koopman aligner-r48`：`accuracy=0.3835`, `RBID=0.3135`
- 这说明：
  - 当前 paper-facing Koopman baseline 仍明显落后于 `EA/RA`
  - 但 `Static Koopman aligner-r48` 的 mismatch 已低于 `Koopman-noalign-r48`
  - 因此 `E1` 最自然的 control 是 `Static Koopman aligner-r48`
- 旧 baseline 缓存与 `E0` 刷新结果存在明显漂移；后续统一以 `E0` 归档结果为准。

## E1 结果（2026-03-09）
- 运行目录：`results/e1/2026-03-09-e1-r1`
- 方法：`Conservative Koopman aligner-r48`
- 相对 `Static Koopman aligner-r48`：
  - `accuracy_mean`: `0.4248 vs 0.3835`，提升 `+0.0413`
  - `rbid`: `0.3056 vs 0.3135`，下降 `-0.0079`
  - `tail_rbid`: `0.5503 vs 0.6357`，下降 `-0.0854`
- 这意味着：
  - 光靠“保守残差对齐 + source 判别保持 + 动力学一致性”，不加 `RBID surrogate`，已经能让当前 Koopman 线明显更稳
  - 当前主 idea 的顺序站住了：先做保守对齐，再把 mismatch-aware supervision 加进去
- 因此 `E2` 的唯一主改动应当是：**在 E1 相同参数化上加入 `RBID surrogate`**，其余保持不变。
