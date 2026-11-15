"""Benchmark wear and quality regressors on held-out grinding wheels."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INPUTS = ["speed_m_s", "feed_mm_s", "depth_um", "minutes"]
TARGETS = ["wear_um", "roughness_um", "abs_error_um"]


def demo():
    rng = np.random.default_rng(23)
    rows = []
    for wheel in range(60):
        wear = 0.0
        susceptibility = rng.lognormal(0, 0.12)
        for run in range(30):
            speed, feed, depth, duration = rng.uniform([22, 2, 8, 1], [38, 8, 28, 4])
            wear += susceptibility * duration * (0.16 + 0.0018 * feed * depth)
            rows.append([wheel, run, speed, feed, depth, duration,
                         max(0, wear + rng.normal(0, 0.8)),
                         0.35 + 0.016*wear + 0.008*depth + rng.normal(0, 0.05),
                         1.0 + 0.07*wear + 0.004*depth**2 + rng.normal(0, 0.35)])
    return pd.DataFrame(rows, columns=["wheel_id", "run", *INPUTS, *TARGETS])


def features(data):
    x = data[INPUTS].copy()
    x["load_proxy"] = x.feed_mm_s * x.depth_um
    x["cumulative_minutes"] = x.minutes.groupby(data.wheel_id).cumsum()
    x["cumulative_load_proxy"] = (x.load_proxy * x.minutes).groupby(data.wheel_id).cumsum()
    x["recent_depth_mean"] = x.depth_um.groupby(data.wheel_id).transform(
        lambda series: series.rolling(5, min_periods=1).mean())
    return x


def analyze(data, out):
    data = data[["wheel_id", "run", *INPUTS, *TARGETS]].copy()
    numeric = ["run", *INPUTS, *TARGETS]
    data[numeric] = data[numeric].astype(float)
    if (data.wheel_id.isna().any() or data.duplicated(["wheel_id", "run"]).any()
            or not np.isfinite(data[numeric]).all().all()
            or (data[numeric] < 0).any().any() or data.wheel_id.nunique() < 10):
        raise ValueError("Supply at least ten wheels with unique ordered runs and finite nonnegative measurements.")
    data = data.sort_values(["wheel_id", "run"]).reset_index(drop=True)
    x = features(data)
    groups = data.wheel_id
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=19).split(x, groups=groups))
    out.mkdir(parents=True, exist_ok=True)
    records = []
    predictions = data.iloc[test][["wheel_id", "run"]].copy()
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
    for index, target in enumerate(TARGETS):
        y = data[target]
        models = {
            "median": DummyRegressor(strategy="median"),
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1)),
            "lightgbm": LGBMRegressor(n_estimators=180, num_leaves=15, learning_rate=0.05,
                                     random_state=19, n_jobs=1, verbosity=-1),
        }
        cv_mae = {
            name: -cross_val_score(model, x.iloc[train], y.iloc[train], groups=groups.iloc[train],
                                   cv=GroupKFold(4), scoring="neg_mean_absolute_error").mean()
            for name, model in models.items()
        }
        selected = min(cv_mae, key=cv_mae.get)
        actual = y.iloc[test]
        for name, model in models.items():
            model.fit(x.iloc[train], y.iloc[train])
            predicted = model.predict(x.iloc[test])
            records.append([target, name, name == selected, cv_mae[name],
                            mean_absolute_error(actual, predicted),
                            mean_squared_error(actual, predicted)**0.5, r2_score(actual, predicted)])
            if name == selected:
                predictions[target + "_actual"] = actual
                predictions[target + "_predicted"] = predicted
        predicted = predictions[target + "_predicted"]
        axis = axes[index, 0]
        axis.scatter(actual, predicted, s=8, alpha=0.5)
        lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        axis.plot([lo, hi], [lo, hi], color="black")
        axis.set(xlabel="Measured (µm)", ylabel="Predicted (µm)", title=f"{target} · {selected}")
        axis = axes[index, 1]
        axis.scatter(predicted, actual - predicted, s=8, alpha=0.5)
        axis.axhline(0, color="black")
        axis.set(xlabel="Predicted (µm)", ylabel="Residual (µm)")
        values = shap.TreeExplainer(models["lightgbm"])(x.iloc[test]).values
        importance = pd.Series(np.abs(values).mean(axis=0), index=x.columns).sort_values()
        importance.to_csv(out / f"shap_{target}.csv", header=["mean_abs_shap"])
        importance.tail(5).plot.barh(ax=axes[index, 2], title="LightGBM · mean absolute SHAP")
    fig.savefig(out / "dashboard.png", dpi=145)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    for i, (wheel, rows) in enumerate(predictions.groupby("wheel_id", sort=False)):
        if i == 5:
            break
        line, = axis.plot(rows.run, rows.wear_um_actual, label=str(wheel))
        axis.plot(rows.run, rows.wear_um_predicted, "--", color=line.get_color())
    axis.set(xlabel="Run", ylabel="Wheel wear (µm)", title="Held-out wheels · solid measured, dashed predicted")
    axis.legend(title="Wheel")
    fig.savefig(out / "wear_trends.png", dpi=145)
    plt.close(fig)
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(records, columns=["response", "model", "selected", "cv_mae", "test_mae",
                                   "test_rmse", "test_r2"]).to_csv(out / "metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV of runs with wheel identifiers and measured responses")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    analyze(pd.read_csv(args.input) if args.input else demo(), args.out)
