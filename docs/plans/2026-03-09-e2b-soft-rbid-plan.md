# E2b Soft-RBID Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current pairwise logistic mismatch surrogate with a target-wise Soft-RBID surrogate while keeping the E2a proxy, RA prior, and evaluation protocol fixed.

**Architecture:** Extend `KoopmanConservativeResidualAligner` with a new `rank_loss_mode` that computes soft ranks from per-source representation scores and minimizes a Huber-smoothed rank gap to fixed behavior ranks. Add a dedicated `E2b` experiment script that reuses the E2a pipeline but swaps only the surrogate mode.

**Tech Stack:** Python, NumPy, SciPy L-BFGS-B, pandas, pytest

---

### Task 1: Add failing unit tests for Soft-RBID primitives

**Files:**
- Modify: `tests/test_conservative_koopman_aligner.py`

**Step 1: Write the failing test**

- Add tests for:
  - rank target normalization on a single target vector;
  - soft ranks preserving score order and staying in `[0, 1]`;
  - soft-rbid loss favoring correctly ordered scores over reversed scores.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_conservative_koopman_aligner.py -q`

**Step 3: Write minimal implementation**

- Add helper methods to `KoopmanConservativeResidualAligner` and/or module-level helpers.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_conservative_koopman_aligner.py -q`

### Task 2: Add failing E2b smoke test

**Files:**
- Create: `tests/test_ksda_e2b.py`

**Step 1: Write the failing test**

- Add parser and single-fold smoke tests for a new `ksda_exp_e2b_soft_rbid.py`.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_ksda_e2b.py -q`

**Step 3: Write minimal implementation**

- Add the new script and required summary/output functions.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_ksda_e2b.py -q`

### Task 3: Implement aligner support for `rank_loss_mode="soft_rbid_huber"`

**Files:**
- Modify: `src/alignment/koopman_alignment.py`
- Test: `tests/test_conservative_koopman_aligner.py`

**Step 1: Implement**

- Add config fields:
  - `rank_loss_mode`
  - `rank_tau`
  - `rank_huber_delta`
- Preserve existing `pairwise_logistic` behavior as default.
- Add the Soft-RBID branch in `fit()` with manual gradient.

**Step 2: Run focused tests**

Run: `./.venv/bin/python -m pytest tests/test_conservative_koopman_aligner.py tests/test_ksda_e2b.py -q`

### Task 4: Implement E2b experiment entrypoint

**Files:**
- Create: `experiments/ksda_exp_e2b_soft_rbid.py`
- Test: `tests/test_ksda_e2b.py`

**Step 1: Implement**

- Reuse E2a loader, evaluation, and reporting shape.
- Change only:
  - method name/key
  - rank loss mode and new hyperparameters
  - controls summary to include E2a as direct surrogate control

**Step 2: Run focused tests**

Run: `./.venv/bin/python -m pytest tests/test_ksda_e2b.py -q`

### Task 5: Run E2b end-to-end and record

**Files:**
- Modify: `docs/plans/2026-03-09-ksda-experiment-ledger.md`
- Modify: `task_plan.md`
- Modify: `notes.md`

**Step 1: Run experiment**

Run: `./.venv/bin/python experiments/ksda_exp_e2b_soft_rbid.py`

**Step 2: Verify outputs**

- Confirm:
  - `results/e2b/<run-tag>/summary.json`
  - `results/e2b/<run-tag>/loso_subject_results.csv`
  - `results/e2b/<run-tag>/pairwise_scores.csv`
  - `results/e2b/<run-tag>/rbid_method_comparison.csv`
  - `results/e2b/<run-tag>/memo.md`

**Step 3: Update records**

- Log main metrics, control deltas, and decision in ledger/notes/task plan.

### Task 6: Full verification

**Files:**
- Modify: none

**Step 1: Run regression verification**

Run: `./.venv/bin/python -m pytest tests/test_koopman_alignment.py tests/test_conservative_koopman_aligner.py tests/test_ksda_e1.py tests/test_ksda_e2.py tests/test_ksda_e2_diagnostic_refresh.py tests/test_ksda_e2_proxy_diagnostic.py tests/test_ksda_e2a.py tests/test_ksda_e2b.py tests/test_rbid.py -q`

**Step 2: Run syntax verification**

Run: `./.venv/bin/python -m py_compile src/alignment/koopman_alignment.py experiments/ksda_exp_e2b_soft_rbid.py`
