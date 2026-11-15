"""Test whether machine-current data improve drill-grinding quality estimates."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORIES = ["GrindingWheel", "Material"]
CONTROLS = ["DoSet", "DcSet", "vf", "vc", "SpiralRise"]
CURRENTS = [channel for family in ("DTRQs", "Is", "Ix", "Iy", "Iz")
            for channel in (f"f1_{family}", f"f2_{family}")]
CORE = [f"DkAct{position}f{flute}" for flute in (1, 2) for position in (2, 21, 40, 55)]


def benchmark(path, out):
    data = pd.read_excel(path)
    data["roughness_um"] = data[["RaS1", "RaS2", "RaN1", "RaN2"]].mean(axis=1, skipna=False)
    data["core_error_um"] = 1000 * data[CORE].sub(data.DcSet, axis=0).abs().mean(axis=1, skipna=False)
    rows, metrics = [], []
    for response in ("roughness_um", "core_error_um"):
        frame = data.dropna(subset=[*CATEGORIES, *CONTROLS, *CURRENTS, response]).reset_index(drop=True)
        x = frame[CATEGORIES + CONTROLS + CURRENTS]
        y = frame[response]
        groups = frame.groupby(CATEGORIES + CONTROLS).ngroup()
        for name, numeric in (("setup_only", CONTROLS), ("setup_plus_currents", CONTROLS + CURRENTS)):
            encoding = ColumnTransformer([
                ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORIES),
                ("continuous", StandardScaler(), numeric),
            ])
            estimator = make_pipeline(encoding, Ridge())
            pred = np.empty(len(y))
            for fold, (train, test) in enumerate(GroupKFold(5).split(x, y, groups), 1):
                search = GridSearchCV(estimator, {"ridge__alpha": [0.1, 1, 10, 100]},
                                      cv=GroupKFold(4), scoring="neg_mean_absolute_error")
                search.fit(x.iloc[train], y.iloc[train], groups=groups.iloc[train])
                pred[test] = np.maximum(0, search.predict(x.iloc[test]))
                rows.extend((frame.name.iloc[i], groups.iloc[i], fold, response, name, y.iloc[i], pred[i])
                            for i in test)
            metrics.append([response, name, len(frame), groups.nunique(),
                            mean_absolute_error(y, pred), r2_score(y, pred)])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["record", "setting", "fold", "response", "model",
                                "measured_um", "predicted_um"]).to_csv(out / "quality_predictions.csv", index=False)
    pd.DataFrame(metrics, columns=["response", "model", "records", "settings", "mae_um", "r2"]).to_csv(
        out / "quality_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("measured-results"))
    args = parser.parse_args()
    benchmark(args.input, args.out)
