"""Viterbi decoding of per-frame H/A/D probabilities into possessions. numpy only, deterministic.

States 0=H, 1=A, 2=D. Staying costs nothing; switching team (H<->A) costs `team_cost`; switching
between alive and dead (H/A<->D) costs `dead_cost`. Costs are in nats (natural log units), so a cost
of 20 means the frames after the switch must, together, be about e^20 times more likely under the
new state before the decoder switches. The costs are chosen on external matches, never on BAR-ATH.
"""

import numpy as np

LABELS = np.array(["H", "A", "D"])


def viterbi(proba, team_cost, dead_cost, eps=1e-6):
    """proba (n, 3) -> int array (n,) of states. Ties break toward the lower state index."""
    n = len(proba)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    logp = np.log(np.clip(proba.astype(np.float64), eps, 1.0))
    trans = np.array([[0.0, team_cost, dead_cost],
                      [team_cost, 0.0, dead_cost],
                      [dead_cost, dead_cost, 0.0]])
    back = np.empty((n, 3), dtype=np.int8)
    score = logp[0].copy()
    for t in range(1, n):
        cand = score[:, None] - trans          # cand[i, j]: come from i into j
        best = np.argmax(cand, axis=0)
        back[t] = best
        score = cand[best, np.arange(3)] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
