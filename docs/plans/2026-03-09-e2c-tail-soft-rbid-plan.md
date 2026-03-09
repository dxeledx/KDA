# E2c Tail-Aware Soft-RBID Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add tail-aware weighting to the existing Soft-RBID surrogate while keeping the E2b proxy, prior, and evaluation protocol fixed.

**Architecture:** Extend the current soft-rank mismatch utilities to support weighted Huber rank-gap loss, with weights derived from low behavior-rank pairs within each target. Reuse the E2b experiment pipeline and change only the rank loss mode and tail-weighting hyperparameters.

**Tech Stack:** Python, NumPy, SciPy L-BFGS-B, pandas, pytest

---

### Task 1: Add failing tail-weight tests

**Files:**
- Modify: `tests/test_conservative_koopman_aligner.py`

**Step 1: Write the failing test**

- Add a test showing tail-weighted Soft-RBID penalizes low-behavior-rank mismatch more than unweighted Soft-RBID.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_conservative_koopman_aligner.py -q`

**Step 3: Write minimal implementation**

- Extend the current Soft-RBID helpers with optional `tail_weight` and `tail_quantile`.

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_conservative_koopman_aligner.py -q`

### Task 2: Add failing E2c smoke test

**Files:**
- Create: `tests/test_ksda_e2c.py`

**Step 1: Write the failing test**

- Add parser and single-fold smoke tests for `ksda_exp_e2c_tail_soft_rbid.py`.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_ksda_e2c.py -q`

### Task 3: Implement tail-aware Soft-RBID in aligner

**Files:**
- Modify: `src/alignment/koopman_alignment.py`
- Test: `tests/test_conservative_koopman_aligner.py`

**Step 1: Implement**

- Add:
  - `rank_tail_weight`
  - `rank_tail_quantile`
  - `rank_loss_mode="tail_soft_rbid_huber"`
- Reuse current soft-rank Jacobian and Huber machinery.
- Weight only low-behavior-rank sources inside each target.

**Step 2: Run focused tests**

Run: `./.venv/bin/python -m pytest tests/test_conservative_koopman_aligner.py tests/test_ksda_e2c.py -q`

### Task 4: Implement E2c experiment entrypoint

**Files:**
- Create: `experiments/ksda_exp_e2c_tail_soft_rbid.py`
- Test: `tests/test_ksda_e2c.py`

**Step 1: Implement**

- Reuse `E2b` pipeline.
- Change only:
  - method name/key
  - rank loss mode and tail-weighting defaults
  - controls summary to include `E2b`

**Step 2: Run focused tests**

Run: `./.venv/bin/python -m pytest tests/test_ksda_e2c.py -q`

### Task 5: Run E2c and record

**Files:**
- Modify: `docs/plans/2026-03-09-ksda-experiment-ledger.md`
- Modify: `task_plan.md`
- Modify: `notes.md`

**Step 1: Run experiment**

Run: `./.venv/bin/python experiments/ksda_exp_e2c_tail_soft_rbid.py`

**Step 2: Record**

- Update ledger/notes/task plan with main metrics and deltas vs `E2b`.

### Task 6: Full verification

**Step 1: Run regression verification**

Run: `./.venv/bin/python -m pytest tests/test_koopman_alignment.py tests/test_conservative_koopman_aligner.py tests/test_ksda_e1.py tests/test_ksda_e2.py tests/test_ksda_e2_diagnostic_refresh.py tests/test_ksda_e2_proxy_diagnostic.py tests/test_ksda_e2a.py tests/test_ksda_e2b.py tests/test_ksda_e2c.py tests/test_rbid.py -q`

**Step 2: Run syntax verification**

Run: `./.venv/bin/python -m py_compile src/alignment/koopman_alignment.py experiments/ksda_exp_e2c_tail_soft_rbid.py`
