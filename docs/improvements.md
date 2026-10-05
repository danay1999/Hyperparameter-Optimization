# Experiment infrastructure improvements

## Investigate parallel-worker shutdown warning

- Recorded: October 4, 2026.
- Run: `bioresponse-noisy-lightgbm-bayesian-100-1-743a2d0ffc0f`.
- Source: joblib's loky `process_executor.py:787`.
- User instruction: retain this for future investigation; do not change the running experiment code now.

Observed warning:

```text
UserWarning: A worker stopped while some jobs were given to the executor.
This can be caused by a too short worker timeout or by a memory leak.
```

The cause is unconfirmed. The warning alone does not establish a memory leak or prove that the experiment failed.

Future investigation:

1. Check the affected run's completion status, trial history, and scores for missing or failed evaluations.
2. Capture worker lifecycle events, process memory usage, system memory pressure, and parallelism settings during a small reproduction.
3. Compare one versus four CV workers on identical configurations and folds; assess repeated pool use and worker idle-timeout behavior.
4. Adjust worker timeout, memory handling, or worker count only after identifying the cause. Preserve visible diagnostics rather than merely suppressing warnings.
5. Log executor warnings alongside run provenance, and test recovery from worker failure without losing completed trial checkpoints.

## Deprecation-warning audit and persistent terminal logs

Recorded: October 4, 2026. Only documentation was changed for this audit.

### Confirmed warnings from previously supplied terminal output

| API / option | Observed warning | Required action / status |
| --- | --- | --- |
| `sklearn.linear_model.LogisticRegression(n_jobs=1)` | `n_jobs` has no effect since 1.8 and will be removed in 1.10. | Constructor updated to omit `n_jobs`. Verify subsequent batches remain warning-free; already running processes retain their imported code. |
| `sklearn.linear_model.LogisticRegression(penalty="elasticnet")` | `penalty` deprecated in 1.8 and will be removed in 1.10; use `l1_ratio` and `C`. | Constructor updated to use `l1_ratio` / `C` for the new API, retaining explicit penalty only for older sklearn versions. Equivalent predictions were checked at mixing ratios 0, 0.5, and 1. |

These were deprecated parameters, not deprecated function names. Their repetition alone does not indicate failed fits. Do not suppress unrelated convergence or executor warnings as part of this cleanup.

### Audit limitation

The live iTerm terminal was not accessible: the computer-use tool explicitly denied access to `com.googlecode.iterm2`. No deprecation messages were found in the available saved `/private/tmp/hpo_tests.txt`, `/private/tmp/hpo_progress.txt`, `/private/tmp/hpo_resume_inventory.txt`, or notebook outputs. This is not a complete review of the active batch's terminal scrollback. Additional warnings remain unverified until the user supplies pasted output or a saved log.

### Future improvements

- Persist both stdout and stderr for each batch, with timestamps, command, package versions, and execution settings, so terminal scrollback is not the only evidence.
- Summarize warnings by category, originating API, message, and affected run; preserve original messages and counts.
- Review supplied additional warnings against the installed package version and official migration guidance before modifying code.
- Add a small classification smoke test that fails on unexpected `FutureWarning` / `DeprecationWarning`; cover the supported sklearn API paths.
- Keep warning-only maintenance separate from scientific protocol changes, and verify predictive equivalence when updating deprecated options.

Reference for the confirmed sklearn changes: [scikit-learn 1.8 release notes](https://scikit-learn.org/stable/whats_new/v1.8.html).
