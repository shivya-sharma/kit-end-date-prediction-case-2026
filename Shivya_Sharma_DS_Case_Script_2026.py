"""Train and validate a kit-end-date regression model for the case study.

Usage:
    python solution.py --data-dir 

The five supplied Parquet files; it keeps the inference
rows in their original order and writes predictions.npy in that order as expected to stimulate a kit cycle time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


DATE_COLUMNS = ["download_date", "kit_start", "kit_end", "est_ship_date"]
CAT_FEATURES = [
    "buid",
    "mcid",
    "lob_desc",
    "family_parent_desc",
    "family_desc",
    "start_weekday",
    "start_month",
]
NUM_FEATURES = [
    "order_amt",
    "quantity_produced",
    "line_qty",
    "cfs_flag",
    "is_cfi",
    "expected_build_minutes",
    "expected_burn_minutes",
    "expected_kit_minutes",
    "expected_kit_load_by_line_minutes",
    "expected_kit_load_by_order_minutes",
    "download_to_start_days",
    "start_day_of_year",
    "start_is_weekend",
    "family_tie_count",
    "cap_ot_flag",
    "cap_ot_hours",
    "cap_regular_hours",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
MISSING = "__MISSING__"
OTHER = "__OTHER__"
ARTIFACT_PREFIX = "Shivya_Sharma_DS_Case"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Downloads" / "Data",
        help="Folder containing train.parquet and the four supplementary files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder where predictions, validation metrics, and mapping are saved.",
    )
    parser.add_argument(
        "--validation-cutoff",
        default="2025-11-01",
        help="Orders whose latest kit_start is on/after this date form the holdout.",
    )
    return parser.parse_args()


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, ...]:
    names = ["train", "inference", "calendar", "capacity", "expected_minutes"]
    tables = []
    for name in names:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
        tables.append(pd.read_parquet(path))
    train, inference, calendar, capacity, expected = tables
    for frame in [train, inference]:
        for col in DATE_COLUMNS:
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
    train["_source_row"] = np.arange(len(train), dtype=np.int64)
    inference["_source_row"] = np.arange(len(inference), dtype=np.int64)
    if inference.duplicated(["sales_order_id", "family_desc"]).any():
        raise ValueError(
            "inference.parquet must be unique by sales_order_id + family_desc "
            "to keep the required one-prediction-per-input-row output."
        )
    return train, inference, calendar, capacity, expected


def prepare_capacity(capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.copy()
    cap["date"] = pd.to_datetime(cap["date"], errors="coerce")
    cap = cap.loc[cap["date"].notna()].copy()
    value_cols = ["ot_flag", "ot_hours", "regular_hours"]
    for col in value_cols:
        cap[col] = pd.to_numeric(cap[col], errors="coerce")
    # The file repeats each date hundreds of times, but its capacity values
    # are identical for every repeated row. Fail rather than silently average
    # if a future version contains conflicting values for the same date.
    conflicts = cap.groupby("date")[value_cols].nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Conflicting capacity rows for {int(conflicts.sum())} dates")
    cap = cap.drop_duplicates("date").rename(
        columns={
            "date": "kit_start",
            "ot_flag": "cap_ot_flag",
            "ot_hours": "cap_ot_hours",
            "regular_hours": "cap_regular_hours",
        }
    )
    return cap[["kit_start", "cap_ot_flag", "cap_ot_hours", "cap_regular_hours"]]


def aggregate_order_family(frame: pd.DataFrame, training: bool) -> pd.DataFrame:
    """Enforce the documented sales_order_id + family_desc training grain."""
    out = frame.copy()
    for col in ["order_amt", "quantity_produced", "line_qty"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if training:
        out["kit_duration_days"] = (out["kit_end"] - out["kit_start"]).dt.days

    key = ["sales_order_id", "family_desc"]
    if not out.duplicated(key, keep=False).any():
        out["family_tie_count"] = 1
        return out

    # The dictionary says this pair is unique, but the supplied train file has
    # repeated family rows with distinct tie numbers. Combining those rows into
    # one family prediction unit to retain the summed line quantity and use the
    # latest tie end date as the family completion date was the apporach I took.
    aggregations: dict[str, Any] = {col: "first" for col in out.columns if col not in key}
    aggregations["line_qty"] = "sum"
    aggregations["kit_start"] = "min"
    aggregations["_source_row"] = "min"
    if training:
        aggregations["kit_end"] = "max"
    grouped = out.groupby(key, dropna=False, as_index=False).agg(aggregations)
    counts = out.groupby(key, dropna=False).size().rename("family_tie_count").reset_index()
    grouped = grouped.drop(columns=["kit_duration_days"], errors="ignore").merge(
        counts, on=key, how="left", validate="1:1"
    )
    if training:
        grouped["kit_duration_days"] = (grouped["kit_end"] - grouped["kit_start"]).dt.days
    return grouped


def enrich_features(
    frame: pd.DataFrame, expected: pd.DataFrame, cap_by_date: pd.DataFrame
) -> pd.DataFrame:
    out = frame.copy()
    exp = expected.copy()
    exp = exp.loc[exp["family_desc"].notna()].copy()
    for col in ["expected_build_minutes", "expected_burn_minutes", "expected_kit_minutes"]:
        exp[col] = pd.to_numeric(exp[col], errors="coerce")
    if exp["family_desc"].duplicated().any():
        raise ValueError("expected_minutes must have one row per family_desc")
    out = out.merge(exp, on="family_desc", how="left", validate="m:1")
    out = out.merge(cap_by_date, on="kit_start", how="left", validate="m:1")

    start = out["kit_start"]
    out["start_weekday"] = start.dt.dayofweek.map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )
    out["start_month"] = start.dt.month.astype("Int64").astype("string")
    out["start_day_of_year"] = start.dt.dayofyear.astype("float64")
    out["start_is_weekend"] = start.dt.dayofweek.isin([5, 6]).astype("int8")
    out["download_to_start_days"] = (start - out["download_date"]).dt.days
    out["expected_kit_load_by_line_minutes"] = (
        out["expected_kit_minutes"] * out["line_qty"]
    )
    out["expected_kit_load_by_order_minutes"] = (
        out["expected_kit_minutes"] * out["quantity_produced"]
    )

    for col in CAT_FEATURES:
        out[col] = out[col].astype("string").fillna(MISSING).astype(str)
    for col in NUM_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fit_category_schema(
    training_features: pd.DataFrame,
) -> tuple[dict[str, list[str]], set[str]]:
    vocabularies: dict[str, list[str]] = {}
    kept_families = set(
        training_features["family_desc"].value_counts().head(250).index.tolist()
    )
    for col in CAT_FEATURES:
        values = training_features[col].astype("string").fillna(MISSING).astype(str)
        if col == "family_desc":
            values = values.where(values.isin(kept_families), OTHER)
        vocab = sorted(set(values.tolist()) | {MISSING, OTHER})
        if len(vocab) > 255:
            # HGB categorical splits are limited to max_bins categories.
            common = values.value_counts().head(253).index.tolist()
            if col == "family_desc":
                kept_families = set(common)
            vocab = sorted(set(common) | {MISSING, OTHER})
        vocabularies[col] = vocab
    return vocabularies, kept_families


def apply_category_schema(
    frame: pd.DataFrame,
    vocabularies: dict[str, list[str]],
    kept_families: set[str],
) -> pd.DataFrame:
    out = frame[FEATURES].copy()
    for col in CAT_FEATURES:
        values = out[col].astype("string").fillna(MISSING).astype(str)
        if col == "family_desc":
            values = values.where(values.isin(kept_families), OTHER)
        valid = set(vocabularies[col])
        values = values.where(values.isin(valid), OTHER)
        out[col] = pd.Categorical(values, categories=vocabularies[col])
    for col in NUM_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=2.0,
        max_bins=255,
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=42,
    )


def make_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=140,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=2.0,
        max_bins=255,
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=42,
    )


def rounded_days(prediction: np.ndarray) -> np.ndarray:
    return np.floor(np.maximum(prediction, 0.0) + 0.5).astype(np.int64)


def date_metrics(actual_days: np.ndarray, predicted_days: np.ndarray) -> dict[str, float]:
    errors = np.abs(actual_days.astype(np.int64) - predicted_days.astype(np.int64))
    return {
        "mae_days": float(errors.mean()),
        "median_absolute_error_days": float(np.median(errors)),
        "rmse_days": float(np.sqrt(np.mean(errors.astype(float) ** 2))),
        "exact_day_accuracy": float(np.mean(errors == 0)),
        "within_one_day_accuracy": float(np.mean(errors <= 1)),
    }


def classification_metrics(
    actual: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    labels = [0, 1, 2]
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "classes": {
            "same_day_0": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(per_class_f1[0]),
                "support": int(support[0]),
            },
            "one_day_1": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(per_class_f1[1]),
                "support": int(support[1]),
            },
            "two_plus_days": {
                "precision": float(precision[2]),
                "recall": float(recall[2]),
                "f1": float(per_class_f1[2]),
                "support": int(support[2]),
            },
        },
        "confusion_matrix_labels_0_1_2plus": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }
    if probabilities is not None:
        try:
            result["macro_ovr_roc_auc"] = float(
                roc_auc_score(actual, probabilities, labels=labels, multi_class="ovr", average="macro")
            )
        except ValueError:
            result["macro_ovr_roc_auc"] = None
    return result


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def save_horizontal_bars(
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    value_labels: list[str],
    axis_label: str,
    color: tuple[int, int, int] = (36, 117, 150),
) -> None:
    width, height = 1500, max(650, 190 + len(labels) * 55)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = load_font(34), load_font(22), load_font(18)
    draw.text((70, 38), title, fill=(25, 39, 52), font=title_font)
    draw.text((70, 92), subtitle, fill=(80, 94, 108), font=small_font)
    left, right = 440, width - 100
    top = 165
    row_height = 55
    plot_width = right - left
    max_value = max(values) if values else 1.0
    max_value = max(max_value * 1.18, 1.0)
    for tick in range(6):
        x = left + int(plot_width * tick / 5)
        draw.line((x, top - 12, x, top + len(labels) * row_height - 5), fill=(224, 231, 237), width=1)
        draw.text((x - 15, top - 40), f"{max_value * tick / 5:.2g}", fill=(105, 115, 124), font=small_font)
    for index, (label, value, value_label) in enumerate(zip(labels, values, value_labels)):
        y = top + index * row_height
        draw.text((70, y + 7), label[:42], fill=(38, 51, 64), font=body_font)
        bar_width = max(1, int(plot_width * value / max_value))
        draw.rounded_rectangle((left, y, left + bar_width, y + 30), radius=6, fill=color)
        draw.text((min(left + bar_width + 12, width - 230), y + 4), value_label, fill=(38, 51, 64), font=small_font)
    draw.text((left, height - 52), axis_label, fill=(80, 94, 108), font=small_font)
    image.save(path, format="PNG", optimize=True)


def save_validation_plot(path: Path, backtests: list[dict[str, Any]]) -> None:
    width, height = 1500, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = load_font(34), load_font(22), load_font(18)
    draw.text((70, 38), "Temporal validation: date MAE", fill=(25, 39, 52), font=title_font)
    draw.text((70, 92), "Expanding training windows; sales orders kept intact across each cutoff", fill=(80, 94, 108), font=small_font)
    left, right, top, bottom = 150, width - 90, 180, height - 160
    y_min, y_max = 0.75, 0.96
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=(224, 231, 237), width=1)
        draw.text((55, y - 12), f"{value:.2f}", fill=(105, 115, 124), font=small_font)
    colors = [(117, 130, 142), (36, 117, 150)]
    keys = ["global_median", "family_parent_median"]
    for color, key in zip(colors, keys):
        points = []
        for index, result in enumerate(backtests):
            x = left + int((right - left) * index / max(1, len(backtests) - 1))
            mae = result[key]["mae_days"]
            y = bottom - int((mae - y_min) / (y_max - y_min) * (bottom - top))
            points.append((x, y))
        draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    labels = [item["cutoff"] for item in backtests]
    for index, label in enumerate(labels):
        x = left + int((right - left) * index / max(1, len(labels) - 1))
        draw.text((x - 45, bottom + 22), label, fill=(65, 77, 89), font=small_font)
    draw.line((left, bottom + 75, left + 55, bottom + 75), fill=colors[0], width=5)
    draw.text((left + 68, bottom + 62), "Global median", fill=(65, 77, 89), font=small_font)
    draw.line((left + 270, bottom + 75, left + 325, bottom + 75), fill=colors[1], width=5)
    draw.text((left + 338, bottom + 62), "Family-parent median", fill=(65, 77, 89), font=small_font)
    image.save(path, format="PNG", optimize=True)


def create_eda_charts(train: pd.DataFrame, backtests: list[dict[str, Any]], output_dir: Path) -> list[str]: 
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    durations = train["kit_duration_days"].astype(int)
    duration_counts = durations.value_counts()
    duration_labels = [str(day) for day in range(9)] + ["9+"]
    duration_values = [int(duration_counts.get(day, 0)) for day in range(9)] + [int(duration_counts[duration_counts.index >= 9].sum())]
    duration_names = [f"{day} days" for day in range(9)] + ["9+ days"]
    duration_text = [f"{val:,} ({val / len(train):.1%})" for val in duration_values]
    path = chart_dir / f"{ARTIFACT_PREFIX}_EDADuration_2026.png"
    save_horizontal_bars(path, "Training kit-cycle duration", "Calendar-day target after excluding invalid negative durations", duration_names, duration_values, duration_text, "Training rows")
    saved.append(path.name)

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    temp = train.assign(start_weekday=train["kit_start"].dt.day_name())
    weekday_stats = temp.groupby("start_weekday", observed=True)["kit_duration_days"].agg(["median", "count"]).reindex(weekday_order).dropna()
    labels = weekday_stats.index.tolist()
    values = weekday_stats["median"].astype(float).tolist()
    value_text = [f"{v:.1f} d  |  n={int(n):,}" for v, n in zip(values, weekday_stats["count"])]
    path = chart_dir / f"{ARTIFACT_PREFIX}_EDAStartWeekday_2026.png"
    save_horizontal_bars(path, "Kit duration by kit-start weekday", "Median calendar days; row counts shown beside bars", labels, values, value_text, "Median kit duration (days)", color=(73, 139, 116))
    saved.append(path.name)

    quantity = pd.to_numeric(train["quantity_produced"], errors="coerce")
    bins = pd.cut(quantity, bins=[-0.1, 1, 2, 4, 8, np.inf], labels=["0–1 units", "2 units", "3–4 units", "5–8 units", "9+ units"], include_lowest=True)
    qty_stats = train.assign(quantity_band=bins).groupby("quantity_band", observed=True)["kit_duration_days"].agg(["median", "count"])
    labels = [str(value) for value in qty_stats.index]
    values = qty_stats["median"].astype(float).tolist()
    value_text = [f"{v:.1f} d  |  n={int(n):,}" for v, n in zip(values, qty_stats["count"])]
    path = chart_dir / f"{ARTIFACT_PREFIX}_EDAQuantity_2026.png"
    save_horizontal_bars(path, "Kit duration by produced quantity", "Product-order quantity bands; median days and sample counts", labels, values, value_text, "Median kit duration (days)", color=(183, 119, 45))
    saved.append(path.name)

    path = chart_dir / f"{ARTIFACT_PREFIX}_ValidationMAE_2026.png"
    save_validation_plot(path, backtests)
    saved.append(path.name)
    return saved


def order_date_metrics(
    validation: pd.DataFrame, predicted_days: np.ndarray
) -> dict[str, float]:
    scored = validation[["sales_order_id", "kit_start", "kit_duration_days"]].copy()
    scored["actual_end"] = scored["kit_start"] + pd.to_timedelta(
        scored["kit_duration_days"], unit="D"
    )
    scored["predicted_end"] = scored["kit_start"] + pd.to_timedelta(predicted_days, unit="D")
    by_order = scored.groupby("sales_order_id", as_index=False).agg(
        actual_end=("actual_end", "max"), predicted_end=("predicted_end", "max")
    )
    actual = (by_order["actual_end"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    predicted = (by_order["predicted_end"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    return date_metrics(actual, predicted)


def rolling_baseline_backtests(train: pd.DataFrame) -> list[dict[str, Any]]:
    """Compare simple, interpretable baselines over ordered holdout windows."""
    results: list[dict[str, Any]] = []
    latest_start = train.groupby("sales_order_id")["kit_start"].max()
    for cutoff_text in [
        "2025-09-01",
        "2025-10-01",
        "2025-11-01",
        "2025-12-01",
        "2026-01-01",
    ]:
        cutoff = pd.Timestamp(cutoff_text)
        heldout_orders = latest_start.index[latest_start.ge(cutoff)]
        fit = train.loc[~train["sales_order_id"].isin(heldout_orders)]
        valid = train.loc[train["sales_order_id"].isin(heldout_orders)]
        global_median = float(fit["kit_duration_days"].median())
        parent_median = fit.groupby("family_parent_desc")["kit_duration_days"].median()
        parent_pred = (
            valid["family_parent_desc"].map(parent_median).fillna(global_median).to_numpy()
        )
        global_pred = np.full(len(valid), global_median)
        y = valid["kit_duration_days"].to_numpy(dtype=np.int64)
        results.append(
            {
                "cutoff": cutoff_text,
                "fit_rows": int(len(fit)),
                "validation_rows": int(len(valid)),
                "validation_sales_orders": int(valid["sales_order_id"].nunique()),
                "global_median": date_metrics(y, rounded_days(global_pred)),
                "family_parent_median": date_metrics(y, rounded_days(parent_pred)),
            }
        )
    return results


def fit_and_predict(args: argparse.Namespace) -> dict[str, Any]:
    train_raw, inference_raw, calendar, capacity, expected = read_inputs(args.data_dir)
    cap_by_date = prepare_capacity(capacity)

    train = aggregate_order_family(train_raw, training=True)
    inference = aggregate_order_family(inference_raw, training=False)
    train = enrich_features(train, expected, cap_by_date)
    inference = enrich_features(inference, expected, cap_by_date)

    # Negative durations are impossible under the stated target definition;
    # retain valid zero-day cycles and preserve legitimate long delays.
    invalid_negative = int((train["kit_duration_days"] < 0).sum())
    missing_target = int(train["kit_duration_days"].isna().sum())
    train = train.loc[
        train["kit_duration_days"].notna() & train["kit_duration_days"].ge(0)
    ].copy()
    train["kit_duration_days"] = train["kit_duration_days"].astype("int64")

    cutoff = pd.Timestamp(args.validation_cutoff)
    order_latest_start = train.groupby("sales_order_id")["kit_start"].max()
    validation_orders = order_latest_start.index[order_latest_start.ge(cutoff)]
    is_validation = train["sales_order_id"].isin(validation_orders)
    fit_rows = train.loc[~is_validation].copy()
    validation = train.loc[is_validation].copy()
    if fit_rows.empty or validation.empty:
        raise ValueError("Temporal validation split is empty; choose another cutoff")

    raw_fit_features = fit_rows[FEATURES]
    raw_validation_features = validation[FEATURES]
    vocab, kept_families = fit_category_schema(raw_fit_features)
    X_fit = apply_category_schema(raw_fit_features, vocab, kept_families)
    X_validation = apply_category_schema(raw_validation_features, vocab, kept_families)
    y_fit = fit_rows["kit_duration_days"].to_numpy(dtype=np.float64)
    y_validation = validation["kit_duration_days"].to_numpy(dtype=np.int64)

    global_median = float(np.median(y_fit))
    baseline_global = rounded_days(np.full(len(validation), global_median))
    family_medians = fit_rows.groupby("family_desc")["kit_duration_days"].median()
    baseline_family = (
        validation["family_desc"].map(family_medians).fillna(global_median).to_numpy()
    )
    baseline_family = rounded_days(baseline_family)
    parent_medians = fit_rows.groupby("family_parent_desc")["kit_duration_days"].median()
    baseline_parent = (
        validation["family_parent_desc"].map(parent_medians).fillna(global_median).to_numpy()
    )
    baseline_parent = rounded_days(baseline_parent)

    model = make_model()
    model.fit(X_fit, y_fit)
    validation_pred_float = model.predict(X_validation)
    validation_pred_days = rounded_days(validation_pred_float)

    # Auxiliary classification to be checked to predict whether the observed duration
    # is same day, one day, or at least two days. This is descriptive primarily;
    # it is not an on-time/late label because the case provided no SLA threshold to me to ingest.
    y_class_fit = np.where(y_fit == 0, 0, np.where(y_fit == 1, 1, 2)).astype("int64")
    y_class_validation = np.where(
        y_validation == 0, 0, np.where(y_validation == 1, 1, 2)
    ).astype("int64")
    classifier = make_classifier()
    classifier.fit(X_fit, y_class_fit)
    class_probabilities = classifier.predict_proba(X_validation)
    class_prediction = classifier.classes_[np.argmax(class_probabilities, axis=1)]
    majority_class = int(pd.Series(y_class_fit).mode().iloc[0])
    majority_prediction = np.full(len(y_class_validation), majority_class, dtype=np.int64)
    classification_evaluation = {
        "target_definition": "0=same-day completion, 1=one calendar day, 2=two or more calendar days",
        "class_prevalence_validation": {
            "same_day_0": float(np.mean(y_class_validation == 0)),
            "one_day_1": float(np.mean(y_class_validation == 1)),
            "two_plus_days": float(np.mean(y_class_validation == 2)),
        },
        "majority_class_baseline": classification_metrics(
            y_class_validation, majority_prediction
        ),
        "hist_gradient_boosting_classifier": classification_metrics(
            y_class_validation, class_prediction, class_probabilities
        ),
        "note": "Auxiliary duration-bucket assessment only; not an SLA classifier.",
    }

    validation_metrics = {
        "cutoff": str(cutoff.date()),
        "fit_rows": int(len(fit_rows)),
        "fit_sales_orders": int(fit_rows["sales_order_id"].nunique()),
        "validation_rows": int(len(validation)),
        "validation_sales_orders": int(validation["sales_order_id"].nunique()),
        "validation_start_min": str(validation["kit_start"].min().date()),
        "validation_start_max": str(validation["kit_start"].max().date()),
        "global_median_baseline": date_metrics(y_validation, baseline_global),
        "family_parent_median_baseline": date_metrics(y_validation, baseline_parent),
        "family_median_baseline": date_metrics(y_validation, baseline_family),
        "hist_gradient_boosting": date_metrics(y_validation, validation_pred_days),
        "sales_order_end_date": {
            "global_median_baseline": order_date_metrics(validation, baseline_global),
            "family_parent_median_baseline": order_date_metrics(validation, baseline_parent),
            "family_median_baseline": order_date_metrics(validation, baseline_family),
            "hist_gradient_boosting": order_date_metrics(validation, validation_pred_days),
        },
    }

    # The hierarchy-median regressoion is the selected model as it beats the
    # complex benchmark on the most recent chronological holdouts, with the
    # global training median as the fallback for unseen product parents as postulated.
    full_parent_medians = train.groupby("family_parent_desc")["kit_duration_days"].median()
    full_global_median = float(train["kit_duration_days"].median())
    predicted_duration_float = (
        inference["family_parent_desc"]
        .map(full_parent_medians)
        .fillna(full_global_median)
        .to_numpy()
    )
    predicted_duration_days = rounded_days(predicted_duration_float)
    predicted_kit_end = (
        inference["kit_start"] + pd.to_timedelta(predicted_duration_days, unit="D")
    ).to_numpy(dtype="datetime64[D]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_npy_path = args.output_dir / f"{ARTIFACT_PREFIX}_Predictions_2026.npy"
    prediction_csv_path = args.output_dir / f"{ARTIFACT_PREFIX}_PredictionRows_2026.csv"
    order_csv_path = args.output_dir / f"{ARTIFACT_PREFIX}_OrderPredictions_2026.csv"
    metrics_path = args.output_dir / f"{ARTIFACT_PREFIX}_Metrics_2026.json"
    np.save(prediction_npy_path, predicted_kit_end)
    prediction_map = inference_raw[
        ["sales_order_id", "family_desc", "kit_start"]
    ].copy()
    prediction_map.insert(0, "inference_row_index", np.arange(len(prediction_map)))
    prediction_map["predicted_kit_duration_days"] = predicted_duration_days
    prediction_map["predicted_kit_end"] = pd.to_datetime(predicted_kit_end)
    prediction_map.to_csv(prediction_csv_path, index=False)
    order_predictions = prediction_map.groupby("sales_order_id", sort=False).agg(
        predicted_sales_order_kit_end=("predicted_kit_end", "max"),
        family_rows=("family_desc", "size"),
    ).reset_index()
    order_predictions.to_csv(order_csv_path, index=False)
    saved_predictions = np.load(prediction_npy_path)
    if len(saved_predictions) != len(inference_raw) or not np.array_equal(
        saved_predictions, predicted_kit_end
    ):
        raise RuntimeError("Saved NumPy predictions failed row-count/order verification")

    backtests = rolling_baseline_backtests(train)
    charts = create_eda_charts(train, backtests, args.output_dir)
    metrics: dict[str, Any] = {
        "data": {
            "training_rows_raw": int(len(train_raw)),
            "training_sales_orders_raw": int(train_raw["sales_order_id"].nunique()),
            "training_rows_after_order_family_aggregation": int(
                len(aggregate_order_family(train_raw, training=True))
            ),
            "training_rows_used": int(len(train)),
            "negative_duration_rows_excluded": invalid_negative,
            "missing_target_rows_excluded": missing_target,
            "training_duration_median_days": float(train["kit_duration_days"].median()),
            "training_duration_mean_days": float(train["kit_duration_days"].mean()),
            "training_duration_p90_days": float(train["kit_duration_days"].quantile(0.90)),
            "training_duration_p99_days": float(train["kit_duration_days"].quantile(0.99)),
            "capacity_input_rows": int(len(capacity)),
            "capacity_unique_dates_after_deduplication": int(len(cap_by_date)),
            "inference_rows": int(len(inference_raw)),
            "inference_sales_orders": int(inference_raw["sales_order_id"].nunique()),
            "inference_rows_with_expected_minutes": int(
                inference["expected_kit_minutes"].notna().sum()
            ),
            "inference_rows_without_expected_minutes": int(
                inference["expected_kit_minutes"].isna().sum()
            ),
            "inference_rows_without_capacity_for_kit_start": int(
                inference["cap_regular_hours"].isna().sum()
            ),
            "inference_start_min": str(inference["kit_start"].min().date()),
            "inference_start_max": str(inference["kit_start"].max().date()),
        },
        "validation": validation_metrics,
        "classification_validation": classification_evaluation,
        "rolling_baseline_backtests": backtests,
        "eda_charts": charts,
        "model": {
            "name": "family_parent_desc median regressor",
            "fallback": "global training median duration",
            "target": "calendar days from kit_start to kit_end",
            "date_rounding": "nearest nonnegative whole calendar day",
            "benchmark": {
                "name": "HistGradientBoostingRegressor",
                "loss": "absolute_error",
                "max_iter": 180,
                "categorical_feature_count": len(CAT_FEATURES),
                "numeric_feature_count": len(NUM_FEATURES),
            },
        },
        "prediction": {
            "rows": int(len(predicted_kit_end)),
            "sales_orders": int(len(order_predictions)),
            "min_predicted_duration_days": int(predicted_duration_days.min()),
            "median_predicted_duration_days": float(np.median(predicted_duration_days)),
            "max_predicted_duration_days": int(predicted_duration_days.max()),
            "duration_frequency": {
                str(k): int(v)
                for k, v in pd.Series(predicted_duration_days).value_counts().sort_index().items()
            },
            "predicted_kit_end_min": str(pd.Timestamp(predicted_kit_end.min()).date()),
            "predicted_kit_end_max": str(pd.Timestamp(predicted_kit_end.max()).date()),
            "npy_order": "same row order as inference.parquet",
        },
        "notes": [
            "calendar.parquet lists site FCJ while train and inference list site KHO; calendar flags were not joined across this mismatch.",
            "capacity.parquet contains repeated date rows with identical values; one row per date was used.",
            "est_ship_date was excluded because it contains 1900-01-01 and pre-kit dates, and its as-of provenance is unclear.",
            "negative kit durations were excluded as invalid; zero-day and long positive cycles were retained.",
            "no SLA threshold is supplied, so no on-time/late classifier was fit.",
            "the hierarchy-median model was selected because it improved date MAE over the global median on the three latest tested cutoffs; the complex benchmark did not improve on the 2025-11-01 holdout.",
        ],
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    fit_and_predict(parse_args())
