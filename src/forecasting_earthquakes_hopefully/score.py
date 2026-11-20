"""How a forecast is scored: one sequence left out at a time. Used by the rivals and by the classifier alike.

A forecast is one number per station-day, larger meaning "more likely". For every group of labels.csv (a
sequence and the days of its region nearest to it) the forecast is made by something that saw only the other
groups; the scores of all the left-out groups together are judged on the days that are not dropped:

  auc          the chance that a positive day scores above a negative day, over all days
  auc_region   the same within each region, then the mean of the regions: knowing only which region is the
               loudest earns nothing here
  recall@a     the alarm is raised where the score is above the level that a share `a` of the training days
               of the fold exceeds; recall is the share of the positive days with the alarm
  alarm@a      the share of the negative days with the alarm (what the alarm costs)
  pairs@a      of the (station, event) pairs, how many have the alarm on at least half of their positive days

A forecast that knows nothing has auc 0.5 and recall equal to alarm.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ALARM_SHARES = (0.05, 0.10, 0.20)


def auc(score, label):
    """Area under the ROC curve, ties counted half; NaN without both classes."""
    label = np.asarray(label, dtype=bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if not n1 or not n0:
        return float("nan")
    return float((rankdata(score)[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def out_of_fold(table, forecast):
    """table: labels.csv with whatever columns the forecast needs. forecast(train, test) -> scores of test.
    Returns a copy of the days that are not dropped with `score` and, per share of ALARM_SHARES, `alarm@a`."""
    use = table[table["dropped"] == 0].reset_index(drop=True)
    out = use.copy()
    out["score"] = np.nan
    for a in ALARM_SHARES:
        out[f"alarm@{a:g}"] = False
    for g in sorted(use["group"].unique()):
        test = use["group"] == g
        train = use[~test]
        s_train = np.asarray(forecast(train, train), dtype=float)
        out.loc[test, "score"] = np.asarray(forecast(train, use[test]), dtype=float)
        for a in ALARM_SHARES:
            out.loc[test, f"alarm@{a:g}"] = out.loc[test, "score"] > np.quantile(s_train, 1 - a)
    return out


def summary(scored):
    """One row of numbers for a table from out_of_fold()."""
    pos, neg = scored[scored["label"] == 1], scored[scored["label"] == 0]
    row = dict(days=len(scored), positive=len(pos), auc=round(auc(scored["score"], scored["label"]), 3),
               auc_region=round(float(np.nanmean([auc(r["score"], r["label"]) for _, r in scored.groupby("region")])), 3))
    pairs = pos.groupby(["station", "event_id"])
    for a in ALARM_SHARES:
        c = f"alarm@{a:g}"
        row[f"recall@{a:g}"] = round(float(pos[c].mean()), 3) if len(pos) else float("nan")
        row[c] = round(float(neg[c].mean()), 3) if len(neg) else float("nan")
        row[f"pairs@{a:g}"] = f"{int((pairs[c].mean() >= 0.5).sum())}/{pairs.ngroups}"
    return row


def by_group(scored, share=0.10):
    """Per left-out group: days, positive days, auc inside the group, recall and alarm share at `share`."""
    c, rows = f"alarm@{share:g}", []
    for g, r in scored.groupby("group"):
        pos, neg = r[r["label"] == 1], r[r["label"] == 0]
        rows.append(dict(group=g, days=len(r), positive=len(pos), auc=round(auc(r["score"], r["label"]), 3),
                         recall=round(float(pos[c].mean()), 3) if len(pos) else float("nan"),
                         alarm=round(float(neg[c].mean()), 3) if len(neg) else float("nan")))
    return pd.DataFrame(rows)
