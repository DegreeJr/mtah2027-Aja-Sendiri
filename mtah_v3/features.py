"""Per-frame features for possession, shared by training (external data) and inference (BAR-ATH).

numpy only. The same function is run on both datasets, so whatever the model learns on the
external matches is computed identically on the competition match.

Inputs for one contiguous block of play (one half), all in metres with the origin at a pitch
corner (x along the length 0..L, y along the width 0..W), NaN where not observed:

    ball   (n, 2)
    home   (n, P, 2)   one column per track slot or player, P can be anything
    away   (n, Q, 2)
    pitch  (L, W)

Output: float32 matrix (n, len(FEATURES)). NaN is allowed; the model handles it.
"""

import warnings

import numpy as np

FPS = 25
CONTROL_RADIUS_M = 2.0     # same evidence rule as starter/possession_v2.py
CONTROL_MARGIN_M = 0.5
NEAR_RADIUS_M = 3.0
WINDOWS = (12, 50, 125)    # half a second, two seconds, five seconds

FEATURES = []


def _register(*names):
    FEATURES.extend(names)


_register("ball_visible", "d_home", "d_away", "d_diff", "n_home_3m", "n_away_3m",
          "ball_speed", "ball_turn", "ball_out_dist", "ball_edge_dist",
          "players_speed", "players_speed_2s", "players_speed_ratio_10s",
          "evidence")
for _w in WINDOWS:
    _register("ev_past_%d" % _w, "ev_future_%d" % _w,
              "vis_past_%d" % _w, "vis_future_%d" % _w,
              "spd_past_%d" % _w, "spd_future_%d" % _w,
              "out_past_%d" % _w, "out_future_%d" % _w)
_register("hold_last_ev", "since_last_ev", "next_ev", "until_next_ev",
          "since_ball_seen", "last_out_dist", "until_ball_seen", "next_out_dist",
          "home_centroid_x_rel", "team_spread_diff")


def _nanmin_last(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(a, axis=-1)


def _rolling_sum(x, w, side):
    """Sum of x over the previous w frames (side='past', excluding current) or next w frames."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    idx = np.arange(n)
    if side == "past":
        lo, hi = np.maximum(0, idx - w), idx
    else:
        lo, hi = idx + 1, np.minimum(n, idx + 1 + w)
    return c[hi] - c[lo], (hi - lo).astype(np.float64)


def _rolling_mean(x, w, side, valid=None):
    """Mean of x over a one-sided window, counting only valid frames. NaN if none valid."""
    if valid is None:
        valid = ~np.isnan(x)
    xs = np.where(valid, x, 0.0)
    s, _ = _rolling_sum(xs, w, side)
    k, _ = _rolling_sum(valid.astype(np.float64), w, side)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, s / np.maximum(k, 1), np.nan)


def _ffill(values, valid):
    """Last valid value at or before each frame, and frames since it (NaN / large if none)."""
    n = len(values)
    idx = np.where(valid, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, values[np.maximum(last, 0)], np.nan)
    since = np.where(last >= 0, np.arange(n) - last, 10 ** 6).astype(np.float64)
    return out, since


def _bfill(values, valid):
    v, s = _ffill(values[::-1], valid[::-1])
    return v[::-1], s[::-1]


def _speed(pos, lag=2):
    """Speed (m/s) from positions lag frames either side; works for (n,2) and (n,P,2)."""
    out = np.full(pos.shape[:-1], np.nan)
    d = np.linalg.norm(pos[2 * lag:] - pos[:-2 * lag], axis=-1) / (2.0 * lag / FPS)
    out[lag:-lag] = d
    return out


def block_features(ball, home, away, pitch):
    L, W = pitch
    n = len(ball)
    ball = ball.astype(np.float64)
    vis = ~np.isnan(ball[:, 0])

    dh_all = np.linalg.norm(home - ball[:, None, :], axis=-1)
    da_all = np.linalg.norm(away - ball[:, None, :], axis=-1)
    d_home = _nanmin_last(dh_all)
    d_away = _nanmin_last(da_all)
    with np.errstate(invalid="ignore"):
        n_home = np.where(vis, np.nansum(dh_all < NEAR_RADIUS_M, axis=1), np.nan)
        n_away = np.where(vis, np.nansum(da_all < NEAR_RADIUS_M, axis=1), np.nan)
        d_diff = d_home - d_away

    bspeed = _speed(ball, lag=1)
    # change of direction: angle between consecutive displacement vectors (radians)
    v1 = ball[1:-1] - ball[:-2]
    v2 = ball[2:] - ball[1:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    turn = np.full(n, np.nan)
    turn[1:-1] = np.arccos(np.clip(cos, -1, 1))

    x, y = ball[:, 0], ball[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out_dist = np.nanmax(np.stack([-x, x - L, -y, y - W], axis=1), axis=1) if n else np.zeros(0)
    out_dist = np.where(vis, out_dist, np.nan)
    edge_dist = np.where(vis, np.minimum(np.minimum(x, L - x), np.minimum(y, W - y)), np.nan)

    allp = np.concatenate([home, away], axis=1)
    pspeed = _speed(allp, lag=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        players_speed = np.nanmean(pspeed, axis=1)
    sp2 = _rolling_mean(players_speed, 25, "past")
    sp10p = _rolling_mean(players_speed, 125, "past")
    sp10f = _rolling_mean(players_speed, 125, "future")
    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sp_ratio = sp2 / np.nanmean(np.stack([sp10p, sp10f]), axis=0)

    nearest = np.fmin(d_home, d_away)
    with np.errstate(invalid="ignore"):
        ok = vis & (nearest <= CONTROL_RADIUS_M) & (np.abs(d_diff) >= CONTROL_MARGIN_M)
    evidence = np.where(ok, np.where(d_home <= d_away, 1.0, -1.0), 0.0)

    cols = [vis.astype(np.float64), d_home, d_away, d_diff, n_home, n_away,
            bspeed, turn, out_dist, edge_dist,
            players_speed, sp2, sp_ratio, evidence]
    is_out = np.where(vis, (out_dist > 0).astype(np.float64), np.nan)
    for w in WINDOWS:
        cols += [_rolling_mean(evidence, w, "past", valid=np.ones(n, bool)),
                 _rolling_mean(evidence, w, "future", valid=np.ones(n, bool)),
                 _rolling_mean(vis.astype(np.float64), w, "past", valid=np.ones(n, bool)),
                 _rolling_mean(vis.astype(np.float64), w, "future", valid=np.ones(n, bool)),
                 _rolling_mean(players_speed, w, "past"),
                 _rolling_mean(players_speed, w, "future"),
                 _rolling_mean(is_out, w, "past"),
                 _rolling_mean(is_out, w, "future")]

    has_ev = evidence != 0
    last_ev, since_ev = _ffill(evidence, has_ev)
    next_ev, until_ev = _bfill(evidence, has_ev)
    _, since_seen = _ffill(np.zeros(n), vis)
    last_out, _ = _ffill(out_dist, vis)
    _, until_seen = _bfill(np.zeros(n), vis)
    next_out, _ = _bfill(out_dist, vis)

    # Team shape: home centroid x relative to away centroid, signed by which end home defends in
    # this block (median over the block), so "home pushed up" has one sign in both halves.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hcx = np.nanmean(home[:, :, 0], axis=1)
        acx = np.nanmean(away[:, :, 0], axis=1)
        rel = hcx - acx
        side = np.sign(np.nanmedian(rel)) if np.isfinite(np.nanmedian(rel)) else 1.0
        side = side if side != 0 else 1.0
        hspread = np.nanstd(home[:, :, 0], axis=1)
        aspread = np.nanstd(away[:, :, 0], axis=1)
    cols += [last_ev, np.minimum(since_ev, 10 ** 4), next_ev, np.minimum(until_ev, 10 ** 4),
             np.minimum(since_seen, 10 ** 4), last_out, np.minimum(until_seen, 10 ** 4), next_out,
             -side * rel, hspread - aspread]

    X = np.stack(cols, axis=1).astype(np.float32)
    assert X.shape[1] == len(FEATURES), (X.shape, len(FEATURES))
    return X
