# Exp-B.1：归一化修正版说明

## 目标

Exp-B.1 只验证一个问题：

> Exp-B 的失败，是否主要来自 `d_src / d_tgt / sigma_recent` 的尺度不匹配？

因此本轮只做：

- source-train 拟合的 feature-wise z-score
- 不改 gate 结构
- 不改 gate 参数
- 不加 `rho_window`
- 不加 feedback

---

## 为什么用 source-fitted z-score

因为这轮想回答的是：

> “是不是只是尺度问题？”

最干净的做法就是：

- 只用 source fold 拟合上下文统计量
- 再把 target trial 的几何上下文映射到同一尺度

这样不会把“尺度修正”和“target 自适应”混在一起。

---

## 为什么现在不做 target-causal normalization

因为 target-causal normalization 会同时引入：

- 目标域历史统计
- 额外的在线自适应

那样即使结果变好，也很难判断到底是：

- feature scaling 修复了问题
还是
- target-side online normalization 本身带来了新收益

当前阶段更需要**归因清晰**，而不是一步做到最强。

---

## 为什么现在不重扫参数

如果这轮同时做：

- 归一化
- bias / temperature / ema 小搜索

那结果会变得难以解释。

Exp-B.1 的设计原则是：

> 先只改输入尺度，看 `w_t` 是否自然变得更有结构。

只有这一步明确有效，才值得做 `Exp-B.2` 的参数微调。
