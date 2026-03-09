# File Tree Sync Design

**Date:** 2026-03-09  
**Status:** approved

## Goal

在仓库根目录维护一个稳定的 `FILE_TREE.md`，用于展示“仓库内容”的完整文件树，并在每次同步到 GitHub 前刷新它。

## Chosen Approach

- 新增生成脚本：`scripts/update_file_tree.py`
- 生成目标文件：`FILE_TREE.md`
- 默认范围：
  - Git 已跟踪文件
  - 非忽略的未跟踪文件
- 默认排除：
  - `.git`
  - `.venv`
  - `__pycache__`
  - `.DS_Store`
  - 其他被 `.gitignore` 忽略的本地噪音

## Why This Approach

- 比手工维护稳定，不会因为忘记更新而失真
- 不依赖本地 Git hook，跨机器更可靠
- 输出稳定、可重复；如果仓库结构没变，重复生成不会制造无意义 diff

## Output Contract

`FILE_TREE.md` 包含：

- 简短说明：生成方式与覆盖范围
- 统计信息：文件数、目录数
- 一个 Markdown 代码块，使用树形结构展示当前仓库内容

## Sync Workflow

后续“同步”流程增加一步：

1. 运行 `python scripts/update_file_tree.py`
2. 检查 `git status`
3. 提交并推送

## Non-Goals

- 不展示所有本地缓存目录
- 不做 Git hook 自动注入
- 不把文件树写进 `README.md`
