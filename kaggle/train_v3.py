"""Train the v3 possession model on EXTERNAL data only (IDSSE, Bassek et al. 2025, CC BY 4.0).

Runs on Kaggle (or anywhere with internet). It never touches the competition match: the BAR-ATH
data stays on our own computer and is only read by starter/possession_v3.py at inference time.

    pip install kloppy==3.18.0 lightgbm==4.6.0
    python kaggle/train_v3.py --out models/v3

Steps
  1. Load the 7 IDSSE matches (Bundesliga 2022/23, 25 fps, official ball-in-play and ball-owning-team
     per frame) with kloppy.
  2. Degrade them to resemble automatic, uncorrected tracking: position noise, player dropouts, and
     a ball that disappears for ~43% of frames, more often in crowded areas and when it is in the
     air. The competition documentation states the ball is visible on 56.9% of tactical frames.
  3. Compute the same per-frame features used at inference (mtah_v3/features.py).
  4. Leave-one-match-out cross-validation: LightGBM on 6 matches, predict the 7th.
  5. Choose the decoder costs (mtah_v3/decode.py) that maximise mean spellF1 over the held-out
     matches, using the competition's own scorer (evaluation/possession_metrics.py).
  6. Train the final model on all 7 matches and write model.txt and config.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from mtah_v3.features import FEATURES, block_features  # noqa: E402
from mtah_v3.decode import LABELS, viterbi  # noqa: E402
import possession_metrics as pm  # noqa: E402

IDSSE_MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
SEED = 20271117
TEAM_COSTS = [5, 10, 20, 40, 80, 160, 320]
DEAD_COSTS = [5, 10, 20, 40, 80, 160, 320]
TRAIN_STRIDE = 2  # neighbouring frames are near-duplicates; every 2nd frame halves training time
TARGET_BALL_MISSING = 0.43
LGB_PARAMS = {
    "objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 200, "feature_fraction": 0.8, "lambda_l2": 1.0,
    "deterministic": True, "force_row_wise": True, "num_threads": 4, "seed": 0, "verbose": -1,
}
NUM_ROUNDS = 300


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def load_idsse(match_id):
    """[{ball (n,2), ball_z (n,), home (n,P,2), away (n,Q,2), y (n,), pitch}] one per period.

    y: 0=H, 1=A, 2=D, -1 = no label (alive but no owning team).
    """
    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False)
    pdim = ds.metadata.pitch_dimensions
    L, W = float(pdim.pitch_length or 105.0), float(pdim.pitch_width or 68.0)
    df = ds.to_df()
    home_team, away_team = ds.metadata.teams[0], ds.metadata.teams[1]

    def cols(team):
        ids = [p.player_id for p in team.players]
        return [i for i in ids if "%s_x" % i in df.columns]
    hid, aid = cols(home_team), cols(away_team)

    bx = df["ball_x"].to_numpy(float)
    by = df["ball_y"].to_numpy(float)
    finite = np.isfinite(bx)
    lo, hi = np.nanmin(bx[finite]), np.nanmax(bx[finite])
    if hi <= 1.5 and lo >= -0.5:          # normalised 0..1
        sx, sy, ox, oy = L, W, 0.0, 0.0
    elif lo < -5:                         # centred metres
        sx, sy, ox, oy = 1.0, 1.0, L / 2, W / 2
    else:                                 # metres from a corner
        sx, sy, ox, oy = 1.0, 1.0, 0.0, 0.0
    print("  %s: %d frames, pitch %.1fx%.1f, %d+%d players, coords scale=%s offset=%s"
          % (match_id, len(df), L, W, len(hid), len(aid), sx, ox))

    def stack(ids):
        x = np.stack([df["%s_x" % i].to_numpy(float) for i in ids], axis=1) * sx + ox
        y = np.stack([df["%s_y" % i].to_numpy(float) for i in ids], axis=1) * sy + oy
        return np.stack([x, y], axis=-1)
    home, away = stack(hid), stack(aid)
    ball = np.stack([bx * sx + ox, by * sy + oy], axis=1)
    ball_z = df["ball_z"].to_numpy(float) if "ball_z" in df.columns else np.zeros(len(df))

    state = df["ball_state"].astype(str).str.lower().to_numpy()
    owner = df["ball_owning_team_id"].astype(str).to_numpy()
    alive = np.array(["alive" in s for s in state])
    y = np.full(len(df), -1, dtype=np.int8)
    y[~alive] = 2
    y[alive & (owner == str(home_team.team_id))] = 0
    y[alive & (owner == str(away_team.team_id))] = 1

    period = df["period_id"].to_numpy()
    blocks = []
    for p in sorted(set(period.tolist())):
        m = period == p
        blocks.append({"ball": ball[m], "ball_z": ball_z[m], "home": home[m], "away": away[m],
                       "y": y[m], "pitch": (L, W), "name": "%s_p%s" % (match_id, p)})
    return blocks


# ---------------------------------------------------------------------------------------------
# Degradation: make clean external tracking look like automatic, uncorrected tracking
# ---------------------------------------------------------------------------------------------

def _chunks(n, start_prob, mean_len, rng, boost=None):
    """Boolean mask of missing frames built from random chunks."""
    p = np.full(n, start_prob) if boost is None else start_prob * boost
    starts = rng.random(n) < p
    lens = np.maximum(1, rng.lognormal(np.log(mean_len), 1.0, n)).astype(int)
    miss = np.zeros(n, dtype=bool)
    f = 0
    while f < n:
        if starts[f]:
            g = min(n, f + lens[f])
            miss[f:g] = True
            f = g
        else:
            f += 1
    return miss


def degrade(block, rng):
    ball = block["ball"].copy()
    home = block["home"].copy()
    away = block["away"].copy()
    n = len(ball)
    home += rng.normal(0, 0.25, home.shape)
    away += rng.normal(0, 0.25, away.shape)
    ball += rng.normal(0, 0.40, ball.shape)
    for arr in (home, away):
        for j in range(arr.shape[1]):
            arr[_chunks(n, 1 / 400, 40, rng), j] = np.nan

    # crowding and height make the ball harder to see
    allp = np.concatenate([block["home"], block["away"]], axis=1)
    with np.errstate(invalid="ignore"):
        crowd = np.nansum(np.linalg.norm(allp - block["ball"][:, None, :], axis=-1) < 5, axis=1)
    boost = 1.0 + 3.0 * (crowd >= 4) + 3.0 * (np.nan_to_num(block["ball_z"]) > 1.5)
    base_missing = np.isnan(block["ball"][:, 0])
    prob = 1 / 150
    for _ in range(6):  # adjust the start rate until the missing share is near the target
        miss = _chunks(n, prob, 30, np.random.default_rng(rng.integers(1 << 31)), boost) | base_missing
        share = miss.mean()
        if abs(share - TARGET_BALL_MISSING) < 0.02:
            break
        prob *= TARGET_BALL_MISSING / max(share, 1e-3)
    ball[miss] = np.nan
    return ball, home, away


# ---------------------------------------------------------------------------------------------
# Scoring with the competition's own scorer
# ---------------------------------------------------------------------------------------------

def spell_f1(y_true, states):
    gt = np.where(y_true >= 0, LABELS[np.clip(y_true, 0, 2)], "X")
    pred = LABELS[states]
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


def decode_blocks(probas, team_cost, dead_cost):
    return [viterbi(p, team_cost, dead_cost) for p in probas]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "v3"))
    ap.add_argument("--synthetic", action="store_true", help="test the pipeline without download")
    args = ap.parse_args()
    import lightgbm as lgb
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    matches = []
    names = IDSSE_MATCHES if not args.synthetic else ["S1", "S2", "S3"]
    for mid in names:
        blocks = load_idsse(mid) if not args.synthetic else synthetic_match(mid, rng)
        out = []
        for b in blocks:
            ball, home, away = degrade(b, rng)
            X = block_features(ball, home, away, b["pitch"])
            out.append({"X": X, "y": b["y"], "name": b["name"],
                        "ball_missing": float(np.isnan(ball[:, 0]).mean())})
        matches.append(out)
        print("  %s ready (%.0fs), ball missing after degradation: %s"
              % (mid, time.time() - t0, ", ".join("%.2f" % o["ball_missing"] for o in out)))

    # leave-one-match-out
    oof = []
    for k in range(len(matches)):
        Xtr = np.concatenate([b["X"] for i, m in enumerate(matches) if i != k for b in m])
        ytr = np.concatenate([b["y"] for i, m in enumerate(matches) if i != k for b in m])
        keep = (ytr >= 0) & (np.arange(len(ytr)) % TRAIN_STRIDE == 0)
        model = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr[keep], ytr[keep], feature_name=FEATURES),
                          num_boost_round=NUM_ROUNDS)
        oof.append([model.predict(b["X"]) for b in matches[k]])
        print("  fold %d/%d trained (%.0fs)" % (k + 1, len(matches), time.time() - t0))

    results = {}
    for tc in TEAM_COSTS:
        for dc in DEAD_COSTS:
            scores = []
            for k, m in enumerate(matches):
                states = np.concatenate(decode_blocks(oof[k], tc, dc))
                y = np.concatenate([b["y"] for b in m])
                scores.append(spell_f1(y, states))
            sf = float(np.mean([s[0] for s in scores]))
            mf = float(np.mean([s[1] for s in scores]))
            results["%d_%d" % (tc, dc)] = {"spellF1": sf, "macroF1": mf,
                                           "per_match_spellF1": [round(s[0], 4) for s in scores]}
            print("  team_cost=%-3d dead_cost=%-3d  spellF1 %.4f  macroF1 %.4f" % (tc, dc, sf, mf))
    # reference points on the same held-out data
    argmax_scores = [spell_f1(np.concatenate([b["y"] for b in m]),
                              np.concatenate([p.argmax(1) for p in oof[k]]))
                     for k, m in enumerate(matches)]
    print("  no decoder (argmax)        spellF1 %.4f" % np.mean([s[0] for s in argmax_scores]))

    best = max(results, key=lambda key: (results[key]["spellF1"], results[key]["macroF1"]))
    tc, dc = (int(v) for v in best.split("_"))
    print("chosen: team_cost=%d dead_cost=%d  spellF1 %.4f" % (tc, dc, results[best]["spellF1"]))

    Xall = np.concatenate([b["X"] for m in matches for b in m])
    yall = np.concatenate([b["y"] for m in matches for b in m])
    keep = (yall >= 0) & (np.arange(len(yall)) % TRAIN_STRIDE == 0)
    final = lgb.train(LGB_PARAMS, lgb.Dataset(Xall[keep], yall[keep], feature_name=FEATURES),
                      num_boost_round=NUM_ROUNDS)
    final.save_model(os.path.join(args.out, "model.txt"))
    config = {
        "team_cost": tc, "dead_cost": dc, "features": FEATURES, "num_rounds": NUM_ROUNDS, "train_stride": TRAIN_STRIDE,
        "lgb_params": LGB_PARAMS, "train_matches": names, "seed": SEED,
        "lightgbm_version": lgb.__version__, "numpy_version": np.__version__,
        "cv": results, "cv_argmax_spellF1": float(np.mean([s[0] for s in argmax_scores])),
        "synthetic": bool(args.synthetic),
    }
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    print("wrote %s/model.txt and config.json (%.0fs)" % (args.out, time.time() - t0))


def synthetic_match(name, rng, n=30000):
    """Tiny stand-in for an IDSSE match, only to test this script end to end."""
    L, W = 105.0, 68.0
    blocks = []
    for p in (1, 2):
        home = rng.uniform([0, 0], [L, W], (n, 11, 2)).cumsum(0) * 0 + rng.uniform([0, 0], [L, W], (1, 11, 2))
        away = rng.uniform([0, 0], [L, W], (1, 11, 2)).repeat(n, 0)
        home = home.repeat(1, 0) + np.cumsum(rng.normal(0, 0.05, (n, 11, 2)), 0)
        away = away + np.cumsum(rng.normal(0, 0.05, (n, 11, 2)), 0)
        y = np.zeros(n, dtype=np.int8)
        team, f = 0, 0
        ball = np.zeros((n, 2))
        while f < n:
            g = min(n, f + int(rng.integers(100, 500)))
            y[f:g] = team if rng.random() > 0.1 else 2
            src = home if team == 0 else away
            ball[f:g] = src[f:g, 0] + rng.normal(0, 0.5, (g - f, 2))
            if y[f] == 2:
                ball[f:g] = (L + 3, W / 2)
            team, f = 1 - team, g
        blocks.append({"ball": ball, "ball_z": np.zeros(n), "home": home, "away": away, "y": y,
                       "pitch": (L, W), "name": "%s_p%d" % (name, p)})
    return blocks


if __name__ == "__main__":
    main()
