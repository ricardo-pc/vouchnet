"""VouchNet: a trust layer for AI agents.

The public API surface (documented in SKILL.md) lives in `main.py`. This
package holds the scoring engine and the tools used to evaluate it:

- `trust`     -- Bayesian shrinkage + TrustRank, the two scoring models
- `detect`    -- graph signals that flag collusion rings and brigading
- `simulate`  -- a deterministic synthetic agent ecosystem with planted fraud
- `evaluate`  -- the harness that measures the models against that ground truth
- `store`     -- review persistence (Supabase, with an in-memory fallback)
"""
