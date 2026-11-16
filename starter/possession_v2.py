"""Team Aja Sendiri, Phase 1 submission: possession as a state that only changes on evidence.

    pip install -r requirements.txt
    python starter/possession_v2.py
    python evaluation/validate_submission.py evaluation/results/aja_sendiri_v2.csv

numpy only, deterministic, no training and no model. Every constant is fixed in PARAMS below,
chosen before the method was run, and justified in the technical document. None of them was chosen
by looking at the public ground truth, the video or any score.

Idea
----
The reference baseline (starter/baseline.py) labels every frame without a visible ball as D, and
then smooths with a majority vote. The ball is missing on about 41% of graded frames, so a
possession gets cut into pieces every time the tracker loses the ball for a couple of seconds.
CHALLENGE.md says the opposite: while the ball is in play one team always has it, and a loose or
contested ball stays with the team that last had it. So this method treats possession as a state:

  1. Evidence. On a frame where the ball is tracked, a team has "control evidence" if its nearest
     player is within CONTROL_RADIUS_M of the ball and clearly closer than the nearest opponent
     (CONTROL_MARGIN_M). Anything else (no ball, ball far from everyone, contested) is no evidence.
  2. Hysteresis. Possession switches to the other team only after it shows SWITCH_FRAMES evidence
     frames in a compact streak (a gap of more than STREAK_GAP_FRAMES resets the streak), and the
     switch is back-dated to the first frame of that streak. A single touch or a deflection does
     not flip possession; missing ball frames keep the current team.
  3. Dead ball, only from explicit signals, never from "the ball is missing":
       - the ball is tracked outside the pitch (by more than OUT_MARGIN_M, for at least
         OUT_MIN_FRAMES frames): dead from that moment until play restarts;
       - a Smart Tagging set piece: dead from the latest dead-ball trigger before it (ball out, or a
         GOALS tag) within a look-back window that depends on the restart type, or, if there is no
         trigger, from the start of the tag; dead until the restart, which is the first ball-speed
         spike inside the tag window (fallback: KICKOFF_TAG_LEAD_FRAMES after the tag start);
       - the second half before its kickoff (the graded block starts before the kickoff).
     After a dead stretch the first team is the team of the player nearest the ball at the restart
     (the taker), or failing that the first team that shows a control streak.
  4. Clean-up. A possession shorter than MIN_SPELL_FRAMES sandwiched between two possessions of the
     other team, with no dead ball in between, is treated as a touch and merged back.
  5. Frames outside the two graded blocks are written as X, which is never scored.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_NAME = "aja_sendiri_v2.csv"

PARAMS = {
    # Control evidence. Vidal-Codina et al. (2022) use a 0.5-1 m possession zone on high-quality
    # tracking and recommend a larger one for lower-quality tracking; this tracking is automatic and
    # uncorrected, so 2 m.
    "CONTROL_RADIUS_M": 2.0,
    # The nearest opponent must be at least this much further away, otherwise it is a duel.
    "CONTROL_MARGIN_M": 0.5,
    # Evidence frames needed to hand possession over: 10 frames = 0.4 s of the ball with one team.
    "SWITCH_FRAMES": 10,
    # Evidence frames further apart than this (1 s) do not belong to the same streak.
    "STREAK_GAP_FRAMES": 25,
    # Ball outside the pitch by more than this, for at least this many frames, means out of play.
    "OUT_MARGIN_M": 1.0,
    "OUT_MIN_FRAMES": 13,
    # A ball-speed spike above this (m/s) inside a set-piece tag is the restart being struck.
    "RESTART_SPEED_MS": 5.0,
    # Restart search window: the tag itself plus this many frames after it.
    "RESTART_SEARCH_PAD": 50,
    # Fallback restart position: starter/01_smart_tagging.ipynb documents that kickoff tags open
    # about 3.5 s (88 frames) before the restart.
    "KICKOFF_TAG_LEAD_FRAMES": 88,
    # How far back from a restart a dead-ball trigger (ball out, goal) may be, per restart type.
    "LOOKBACK_FRAMES": {
        "throw-in": 25 * 30,
        "corner kick": 25 * 45,
        "goal kick": 25 * 45,
        "free kick": 25 * 60,
        "kick off (after goal)": 25 * 150,
        "kick off (start)": 0,
    },
    # Cap on a dead stretch started by the ball going out when no set piece is tagged after it.
    "OUT_UNTAGGED_MAX_FRAMES": 25 * 30,
    # Possessions shorter than this between two possessions of the other team are touches.
    "MIN_SPELL_FRAMES": 50,
}

HOME, AWAY, NONE = 0, 1, -1
FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"


def read_tracking(raw_path, team_of_track, n_frames, pitch):
    """Ball position (m) and nearest home / away player distance (m), per video frame. NaN = none."""
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan)
    near = np.full((n_frames, 2), np.inf)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if not 0 <= frame < n_frames:
                continue
            b = parts[2]
            if not b or b[0] == MISSING:
                continue
            fields = b.split(",")
            bx, by = float(fields[0]) * length, float(fields[1]) * width
            ball[frame] = (bx, by)
            best = [np.inf, np.inf]
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING or slot >= len(team_of_track):
                    continue
                fields = group.split(",")
                dx = float(fields[0]) * length - bx
                dy = float(fields[1]) * width - by
                d = (dx * dx + dy * dy) ** 0.5
                team = team_of_track[slot]
                if d < best[team]:
                    best[team] = d
            near[frame] = best
    return ball, near


def control_evidence(near, p):
    """HOME / AWAY where one team clearly controls the ball, NONE elsewhere."""
    nearest = np.minimum(near[:, 0], near[:, 1])
    team = np.where(near[:, 0] <= near[:, 1], HOME, AWAY)
    with np.errstate(invalid="ignore"):
        gap = np.abs(near[:, 0] - near[:, 1])  # inf - inf (ball seen, nobody tracked) is NaN
    ok = (nearest <= p["CONTROL_RADIUS_M"]) & (gap >= p["CONTROL_MARGIN_M"])
    return np.where(ok, team, NONE).astype(np.int8)


def ball_speed(ball):
    """Ball speed in m/s from positions two frames apart; NaN where either end is missing."""
    speed = np.full(len(ball), np.nan)
    step = np.linalg.norm(ball[2:] - ball[:-2], axis=1) / (2.0 / 25.0)
    speed[1:-1] = step
    return speed


def out_of_play(ball, pitch, p):
    """Boolean per frame: ball tracked outside the pitch for a sustained run."""
    length, width = pitch
    m = p["OUT_MARGIN_M"]
    x, y = ball[:, 0], ball[:, 1]
    seen = ~np.isnan(x)
    out = np.zeros(len(ball), dtype=bool)
    out[seen] = ((x[seen] < -m) | (x[seen] > length + m) | (y[seen] < -m) | (y[seen] > width + m))
    # keep only runs of at least OUT_MIN_FRAMES
    keep = np.zeros_like(out)
    f, n = 0, len(out)
    while f < n:
        if out[f]:
            g = f
            while g < n and out[g]:
                g += 1
            if g - f >= p["OUT_MIN_FRAMES"]:
                keep[f:g] = True
            f = g
        else:
            f += 1
    return keep


def restart_frame(tag, speed, p, n):
    """First ball-speed spike inside the tag window, else the documented fallback."""
    lo = max(0, tag["start_frame"])
    hi = min(n, tag["end_frame"] + p["RESTART_SEARCH_PAD"])
    window = speed[lo:hi]
    hits = np.flatnonzero(np.nan_to_num(window, nan=0.0) > p["RESTART_SPEED_MS"])
    if len(hits):
        return lo + int(hits[0])
    return min(n - 1, lo + p["KICKOFF_TAG_LEAD_FRAMES"])


def dead_ball(n, ball, speed, tags, pitch, blocks, evidence, p):
    """Boolean per frame: ball out of play. Also returns the list of restart frames."""
    dead = np.zeros(n, dtype=bool)
    out = out_of_play(ball, pitch, p)
    restarts = []

    triggers = np.zeros(n, dtype=bool)
    triggers |= out
    for tag in tags:
        if (tag["code"] or "").strip().lower() == "goals":
            triggers[max(0, tag["start_frame"]):min(n, tag["end_frame"])] = True

    second_half_start = blocks[1][1] if len(blocks) > 1 else None
    for tag in tags:
        if (tag["code"] or "").strip().lower() != "set pieces":
            continue
        kind = (tag["tags"].get("Type") or "").strip().lower()
        restart = restart_frame(tag, speed, p, n)
        restarts.append(restart)
        if kind == "kick off (start)":
            # The second half's graded block starts before its kickoff; the first half's does not.
            if second_half_start is not None and second_half_start <= restart:
                if restart - second_half_start < 25 * 600:
                    dead[second_half_start:restart] = True
            continue
        start = min(max(0, tag["start_frame"]), restart)
        lookback = p["LOOKBACK_FRAMES"].get(kind, 25 * 30)
        lo = max(0, restart - lookback)
        hit = np.flatnonzero(triggers[lo:restart])
        if len(hit):
            # walk back to the first frame of that trigger run
            first = lo + int(hit[-1])
            while first > lo and triggers[first - 1]:
                first -= 1
            start = min(start, first)
        dead[start:restart] = True

    # Ball out with no tagged restart: dead until the next frame with control evidence (at most
    # OUT_UNTAGGED_MAX_FRAMES, so a tracking glitch cannot swallow a long stretch of play).
    for lo, hi in segments(out):
        if dead[lo]:
            continue
        cap = min(n, hi + p["OUT_UNTAGGED_MAX_FRAMES"])
        after = np.flatnonzero(evidence[hi:cap] != NONE)
        end = hi + int(after[0]) if len(after) else cap
        dead[lo:end] = True
    return dead, out, sorted(restarts)


def segments(mask):
    """[(start, end)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def possession_states(evidence, live_lo, live_hi, first_team, p):
    """Hysteresis over one live stretch. Returns an int8 array of HOME/AWAY of length hi-lo."""
    k, gap = p["SWITCH_FRAMES"], p["STREAK_GAP_FRAMES"]
    out = np.empty(live_hi - live_lo, dtype=np.int8)
    state = first_team
    streak_team, streak_start, streak_count, last_seen = NONE, -1, 0, -10 ** 9
    last_written = live_lo
    for f in range(live_lo, live_hi):
        e = evidence[f]
        if e == NONE:
            continue
        if e == state:
            streak_team, streak_count = NONE, 0
            continue
        if e != streak_team or f - last_seen > gap:
            streak_team, streak_start, streak_count = e, f, 0
        streak_count += 1
        last_seen = f
        if streak_count >= k:
            out[last_written - live_lo:streak_start - live_lo] = state
            last_written = streak_start
            state = e
            streak_team, streak_count = NONE, 0
    out[last_written - live_lo:] = state
    return out


def first_team_after(f, hi, evidence, near, ball, prev_team, p):
    """Team that starts a live stretch: the taker at the restart, else the first control streak."""
    if f < len(near) and not np.isnan(ball[f, 0]) and np.isfinite(near[f]).any():
        return HOME if near[f, 0] <= near[f, 1] else AWAY
    k, gap = p["SWITCH_FRAMES"], p["STREAK_GAP_FRAMES"]
    team, count, last = NONE, 0, -10 ** 9
    for g in range(f, hi):
        e = evidence[g]
        if e == NONE:
            continue
        if e != team or g - last > gap:
            team, count = e, 0
        count += 1
        last = g
        if count >= k:
            return team
    return prev_team if prev_team != NONE else HOME


def merge_short_spells(labels, p):
    """X Y X with |Y| < MIN_SPELL_FRAMES and no D between: Y becomes X."""
    runs = []
    f, n = 0, len(labels)
    while f < n:
        g = f
        while g < n and labels[g] == labels[f]:
            g += 1
        runs.append([labels[f], f, g])
        f = g
    for i in range(1, len(runs) - 1):
        lab, lo, hi = runs[i]
        before, after = runs[i - 1][0], runs[i + 1][0]
        if lab in ("H", "A") and before == after and before in ("H", "A") and before != lab \
                and hi - lo < p["MIN_SPELL_FRAMES"]:
            labels[lo:hi] = before
            runs[i][0] = before
    return labels


def build(data_dir=None, game=GAME, p=PARAMS):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading tracking ...")
    ball, near = read_tracking(raw_path, team_of_each_track(meta_path), n, pitch)
    evidence = control_evidence(near, p)
    speed = ball_speed(ball)
    tags = load_smart_tagging.load(game, data_dir)
    blocks = half_video_blocks(game)

    dead, _, _ = dead_ball(n, ball, speed, tags, pitch, blocks, evidence, p)

    labels = np.full(n, "X", dtype="<U1")
    prev_team = NONE
    for half, lo, hi in blocks:
        hi = hi + 1  # blocks are inclusive
        block_dead = dead[lo:hi]
        labels[lo:hi][block_dead] = "D"
        live = segments(~block_dead)
        for s, e in live:
            s, e = lo + s, lo + e
            first = first_team_after(s, e, evidence, near, ball, prev_team, p)
            states = possession_states(evidence, s, e, first, p)
            labels[s:e] = np.where(states == HOME, "H", "A")
            prev_team = states[-1]
    labels = merge_short_spells(labels, p)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, OUT_NAME))
    args = parser.parse_args()
    labels = build()
    path = write_submission(args.out, labels)
    print("wrote {}".format(path))
    print("next: python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))
