"""Nested held-out experiment benchmark for measured insert flank wear."""
import argparse
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.io import loadmat
from scipy.stats import kurtosis
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHANNELS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
PROCESS = ["run", "time", "DOC", "feed", "material"]


def extract(path):
    with ZipFile(path) as archive, archive.open("mill.mat") as mat:
        records = loadmat(mat, simplify_cells=True)["mill"]
    rows = []
    for record in records:
        if not np.isfinite(record["VB"]):
            continue
        row = {key: record[key] for key in ["case", *PROCESS]}
        row["wear_mm"] = record["VB"]
        for key in CHANNELS:
            raw = np.asarray(record[key], dtype=float).ravel()
            z = raw[len(raw)//4:3*len(raw)//4]
            if len(z) < 4 or not np.isfinite(z).all():
                continue
            p10, p50, p90 = np.quantile(z, [0.1, 0.5, 0.9])
            centered = z - z.mean()
            spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(z))))**2
            energy = spectrum.sum()
            row.update({
                f"{key}_median": p50,
                f"{key}_spread": p90 - p10,
                f"{key}_rms": np.sqrt(np.mean(z*z)),
                f"{key}_kurtosis": kurtosis(z) if np.std(z) > 1e-12 else 0,
                f"{key}_upper_band_fraction": spectrum[len(spectrum)//2:].sum()/energy if energy else 0,
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["case", "run"]).reset_index(drop=True)


def benchmark(data, out):
    target, groups = data.wear_mm, data.case
    sensors = [key for key in data if key not in ["case", "wear_mm", *PROCESS]]
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
    tree = LGBMRegressor(n_estimators=120, num_leaves=7, learning_rate=0.04, n_jobs=1,
                         min_child_samples=8, random_state=73, verbosity=-1)
    models = [
        ("median", PROCESS, DummyRegressor(strategy="median"), {}),
        ("process_ridge", PROCESS, ridge, {"ridge__alpha": [0.1, 1, 10, 100]}),
        ("sensor_ridge", sensors, ridge, {"ridge__alpha": [0.1, 1, 10, 100]}),
        ("fused_ridge", PROCESS + sensors, ridge, {"ridge__alpha": [0.1, 1, 10, 100]}),
        ("fused_lightgbm", PROCESS + sensors, tree, {"min_child_samples": [5, 12, 20]}),
    ]
    rows = []
    for train, test in LeaveOneGroupOut().split(data, target, groups):
        fitted = []
        for name, columns, est, grid in models:
            search = GridSearchCV(est, grid, cv=GroupKFold(4), scoring="neg_mean_absolute_error")
            search.fit(data.iloc[train][columns], target.iloc[train], groups=groups.iloc[train])
            pred = np.maximum(0, search.predict(data.iloc[test][columns]))
            fitted.append((search.best_score_, name, pred))
        _, winner, best = max(fitted, key=lambda item: item[0])
        for _, name, pred in [*fitted, (0, "selected", best)]:
            rows.extend(zip(groups.iloc[test], data.run.iloc[test], [name]*len(test),
                            [winner]*len(test), target.iloc[test], pred))

    out.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(rows, columns=["case", "run", "model", "fold_winner", "measured_mm", "predicted_mm"])
    predictions["absolute_error_mm"] = (predictions.measured_mm - predictions.predicted_mm).abs()
    case_mae = predictions.groupby(["model", "case"]).absolute_error_mm.mean().unstack()
    case_mae.to_csv(out / "case_mae.csv")
    rng = np.random.default_rng(73)
    bootstrap = rng.integers(case_mae.shape[1], size=(4000, case_mae.shape[1]))
    metrics = []
    for model, subset in predictions.groupby("model"):
        a, b = subset.measured_mm, subset.predicted_mm
        cases = case_mae.loc[model].to_numpy()
        lo, hi = np.quantile(cases[bootstrap].mean(axis=1), [0.025, 0.975])
        metrics.append([model, mean_absolute_error(a, b), np.sqrt(mean_squared_error(a, b)),
                        r2_score(a, b), cases.mean(), lo, hi])
    pd.DataFrame(metrics, columns=["model", "pooled_mae_mm", "pooled_rmse_mm", "pooled_r2",
                                   "case_macro_mae_mm", "case_bootstrap_lo", "case_bootstrap_hi"]).to_csv(
                                       out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    selected = predictions[predictions.model == "selected"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for case, subset in selected.groupby("case"):
        axes[0].scatter(subset.measured_mm, subset.predicted_mm, s=17, alpha=0.65)
    hi = selected[["measured_mm", "predicted_mm"]].max().max()
    axes[0].plot([0, hi], [0, hi], color="#1d2935")
    axes[0].set(xlabel="Measured flank wear VB (mm)", ylabel="Predicted VB (mm)", title="Held-out milling experiments")
    errors = case_mae.loc["selected"]
    axes[1].bar(errors.index.astype(str), errors, color="#336b89")
    axes[1].set(xlabel="Held-out experiment", ylabel="MAE (mm)", title="Error by experiment")
    fig.savefig(out / "diagnostics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("mill.zip"))
    parser.add_argument("--out", type=Path, default=Path("measured-results"))
    args = parser.parse_args()
    data = extract(args.input)
    print(f"{len(data)} measured wear records in {data.case.nunique()} cases")
    benchmark(data, args.out)
