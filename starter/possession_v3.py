"""Team Aja Sendiri, v3: a model trained on external matches, decoded into possessions.

    pip install -r requirements.txt
    python starter/possession_v3.py
    python evaluation/validate_submission.py evaluation/results/aja_sendiri_v3.csv

The model (models/v3/model.txt) and the decoder costs (models/v3/config.json) were produced by
kaggle/train_v3.py from the IDSSE open dataset (Bassek et al. 2025, CC BY 4.0) only. Nothing in
them was fitted, tuned or chosen on the competition match, its ground truth or its video.

Here, on our own computer, the BAR-ATH tracking is read, the same features are computed
(mtah_v3/features.py), the model gives H/A/D probabilities per frame, and a Viterbi decoder turns
them into possessions (mtah_v3/decode.py). Frames outside the two graded halves are X. Inference is
on CPU and has no randomness, so the file is reproduced byte for byte.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names, write_submission  # noqa: E402
from mtah_v3.features import FEATURES, block_features  # noqa: E402
from mtah_v3.decode import LABELS, viterbi  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "v3")
OUT = os.path.join(ROOT, "evaluation", "results", "aja_sendiri_v3.csv")
FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"


def read_positions(raw_path, n_frames, n_slots, pitch):
    """Ball (n,2) and every track slot (n, slots, 2), in metres from a pitch corner. NaN = unseen."""
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan)
    slots = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            b = parts[2]
            if b and b[0] != MISSING:
                fields = b.split(",")
                ball[frame] = (float(fields[0]) * length, float(fields[1]) * width)
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots or not group or group[0] == MISSING:
                    continue
                fields = group.split(",")
                slots[frame, slot] = (float(fields[0]) * length, float(fields[1]) * width)
    return ball, slots


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import lightgbm as lgb
    with open(os.path.join(args.model, "config.json")) as fh:
        config = json.load(fh)
    if config["features"] != FEATURES:
        raise SystemExit("model was trained on a different feature list; retrain it")
    booster = lgb.Booster(model_file=os.path.join(args.model, "model.txt"))

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    team = team_of_each_track(meta_path)
    print("teams: H = {}, A = {}".format(*team_names(meta_path)))
    print("reading tracking ...")
    ball, slots = read_positions(raw_path, n, len(team), pitch)
    home, away = slots[:, team == 0], slots[:, team == 1]

    labels = np.full(n, "X", dtype="<U1")
    for half, lo, hi in half_video_blocks(GAME):
        hi += 1  # inclusive block
        X = block_features(ball[lo:hi], home[lo:hi], away[lo:hi], pitch)
        proba = booster.predict(X, num_threads=1)
        states = viterbi(proba, config["team_cost"], config["dead_cost"])
        labels[lo:hi] = LABELS[states]
        print("  half {} decoded".format(half))
    path = write_submission(args.out, labels)
    print("wrote {}".format(path))


if __name__ == "__main__":
    main()
