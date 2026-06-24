#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/osda_lazyfca_matplotlib")

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    raise SystemExit(
        "Missing dependency: PyYAML. Install project requirements or run: "
        "python3 -m pip install PyYAML"
    ) from exc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.compose
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

try:
    from lazyfca import LazyFCA
    from lazyfca.metrics import METADATA_DICT
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    if exc.name == "numba":
        raise SystemExit(
            "Missing dependency: numba. Install project requirements before running LazyFCA experiments."
        ) from exc
    raise


PRIMARY_METRIC = "primary_f1"
VANILLA_METRIC = "all"
RANDOM_METRIC = "random"


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    target: str
    drop_columns: tuple[str, ...] = ()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping at top level: {path}")
    return config


def enabled_names(section: dict) -> list[str]:
    names = []
    for name, value in section.items():
        enabled = True
        if isinstance(value, dict):
            enabled = bool(value.get("enabled", True))
        elif isinstance(value, bool):
            enabled = value
        if enabled:
            names.append(name)
    return names


def selected_names(config_names: list[str], cli_names: typing.Optional[list[str]]) -> list[str]:
    if not cli_names:
        return config_names
    requested = set(cli_names)
    missing = sorted(requested.difference(config_names))
    if missing:
        raise ValueError(f"Requested unknown or disabled names: {missing}")
    return [name for name in config_names if name in requested]


def dataset_specs(config: dict, names: list[str]) -> list[DatasetSpec]:
    specs = []
    for name in names:
        raw = config["datasets"][name]
        specs.append(
            DatasetSpec(
                name=name,
                path=ROOT / raw["path"],
                target=raw["target"],
                drop_columns=tuple(raw.get("drop_columns", []) or []),
            )
        )
    return specs


def one_hot_encoder():
    try:
        return sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", dtype=bool, sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", dtype=bool, sparse=False)


def preprocess_dataset(spec: DatasetSpec, test_size: float, seed: int):
    df = pd.read_csv(spec.path)
    missing = [col for col in [spec.target, *spec.drop_columns] if col not in df.columns]
    if missing:
        raise ValueError(f"{spec.name}: missing columns {missing}")

    y_raw = df[spec.target]
    X = df.drop(columns=[spec.target, *spec.drop_columns])
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train_raw, X_test_raw, y_train, y_test = sklearn.model_selection.train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    numeric_cols = X_train_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X_train_raw.columns if col not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", one_hot_encoder(), categorical_cols))

    if transformers:
        preprocessor = sklearn.compose.ColumnTransformer(transformers=transformers)
        X_train_arr = preprocessor.fit_transform(X_train_raw)
        X_test_arr = preprocessor.transform(X_test_raw)
        columns = preprocessor.get_feature_names_out()
        X_train = pd.DataFrame(X_train_arr, columns=columns, index=X_train_raw.index)
        X_test = pd.DataFrame(X_test_arr, columns=columns, index=X_test_raw.index)
    else:
        X_train = pd.DataFrame(index=X_train_raw.index)
        X_test = pd.DataFrame(index=X_test_raw.index)

    categorical_features = [col for col in X_train.columns if col.startswith("categorical__")]
    if categorical_features:
        X_train[categorical_features] = X_train[categorical_features].astype(bool)
        X_test[categorical_features] = X_test[categorical_features].astype(bool)

    for col in X_train.columns:
        if col not in categorical_features:
            X_train[col] = pd.to_numeric(X_train[col], errors="raise").astype(float)
            X_test[col] = pd.to_numeric(X_test[col], errors="raise").astype(float)

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": pd.Series(y_train).reset_index(drop=True),
        "y_test": np.asarray(y_test, dtype=int),
        "label_names": [str(label) for label in label_encoder.classes_],
        "n_classes": int(len(label_encoder.classes_)),
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "encoded_feature_count": int(X_train.shape[1]),
    }


def stable_json(data: typing.Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(data: typing.Any) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()[:12]


def dataset_fingerprint(spec: DatasetSpec, test_size: float, seed: int) -> str:
    stat = spec.path.stat()
    return short_hash(
        {
            "name": spec.name,
            "path": str(spec.path.relative_to(ROOT)),
            "path_size": stat.st_size,
            "path_mtime_ns": stat.st_mtime_ns,
            "target": spec.target,
            "drop_columns": spec.drop_columns,
            "test_size": test_size,
            "seed": seed,
        }
    )


def chunk_path(run_dir: Path, dataset: str, seed: int, method: str, metric: str) -> Path:
    return run_dir / "chunks" / f"{dataset}__seed{seed}__{method}__{metric}.csv"


def diagnostic_path(run_dir: Path, dataset: str, seed: int) -> Path:
    return run_dir / "diagnostics" / f"{dataset}__seed{seed}.csv"


def append_manifest(run_dir: Path, event: dict) -> None:
    path = run_dir / "manifest.jsonl"
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def prepare_dataset(config: dict, spec: DatasetSpec, seed: int) -> dict:
    data = preprocess_dataset(spec, float(config["test_size"]), seed)
    model = LazyFCA(n_classes=data["n_classes"])
    model.fit(data["X_train"], data["y_train"])

    return {
        "dataset": spec.name,
        "seed": seed,
        "test_size": float(config["test_size"]),
        "model": model,
        "X_test": data["X_test"],
        "X_train_shape": tuple(data["X_train"].shape),
        "X_test_shape": tuple(data["X_test"].shape),
        "y_train": data["y_train"].to_numpy(dtype=int),
        "y_test": data["y_test"],
        "label_names": data["label_names"],
        "n_classes": data["n_classes"],
        "numeric_feature_count": data["numeric_feature_count"],
        "categorical_feature_count": data["categorical_feature_count"],
        "encoded_feature_count": data["encoded_feature_count"],
        "class_priors": np.bincount(data["y_train"], minlength=data["n_classes"]).astype(float)
        / float(len(data["y_train"])),
    }


def flatten_classifiers(explanation) -> list[tuple[int, int, typing.Any]]:
    flat = []
    for class_index, classifiers in enumerate(explanation.class_classifiers):
        for source_index, classifier in enumerate(classifiers):
            flat.append((class_index, source_index, classifier))
    return flat


def safe_score(classifier, metric: str) -> float:
    value = classifier.metrics.score_for_ranking(metric)
    if value is None or np.isnan(value):
        return float("-inf")
    return float(value)


def ranked_classifiers(explanation, metric: str) -> list[tuple[int, int, typing.Any, float]]:
    rows = []
    for class_index, source_index, classifier in flatten_classifiers(explanation):
        score = safe_score(classifier, metric)
        query_similarity = classifier.metrics.get_metric("query_similarity")
        tp = classifier.metrics.get_metric("tp")
        fp = classifier.metrics.get_metric("fp")
        rows.append((class_index, source_index, classifier, score, query_similarity, tp, fp))

    rows.sort(
        key=lambda row: (
            -row[3],
            -(float(row[4]) if row[4] is not None and np.isfinite(row[4]) else float("-inf")),
            -(int(row[5]) if row[5] is not None else -1),
            int(row[6]) if row[6] is not None else 10**12,
            row[0],
            row[1],
        )
    )
    return [(class_index, source_index, classifier, score) for class_index, source_index, classifier, score, *_ in rows]


def k_values_from_config(k_grid: dict, full_k: int, smoke: bool) -> list[int]:
    if smoke:
        return [k for k in [1, 2, 5] if k <= full_k]

    values = [int(k) for k in k_grid.get("low", []) if int(k) >= 1]
    geometric = k_grid.get("geometric", {}) or {}
    start = int(geometric.get("start", 12))
    stop_raw = geometric.get("stop", "full")
    stop = full_k if stop_raw == "full" else int(stop_raw)
    num = int(geometric.get("num", 24))
    if full_k >= start and num > 0:
        values.extend(np.geomspace(start, min(stop, full_k), num=num).round().astype(int).tolist())
    values.append(full_k)
    return sorted(set(k for k in values if 1 <= k <= full_k))


def choose_class(counts: np.ndarray, score_sums: np.ndarray, priors: np.ndarray) -> int:
    tied = np.flatnonzero(counts == counts.max())
    if len(tied) == 1:
        return int(tied[0])
    tied_scores = score_sums[tied]
    tied = tied[np.flatnonzero(tied_scores == tied_scores.max())]
    if len(tied) == 1:
        return int(tied[0])
    tied_priors = priors[tied]
    tied = tied[np.flatnonzero(tied_priors == tied_priors.max())]
    return int(tied.min())


def predict_from_retained(
    retained: list[tuple[int, int, typing.Any, float]],
    n_classes: int,
    priors: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(n_classes, dtype=float)
    score_sums = np.zeros(n_classes, dtype=float)
    for class_index, _source_index, _classifier, score in retained:
        counts[class_index] += 1.0
        if np.isposinf(score):
            score_sums[class_index] = np.inf
        elif np.isfinite(score):
            score_sums[class_index] += score
    pred = choose_class(counts, score_sums, priors)
    proba = counts / counts.sum() if counts.sum() > 0 else priors.copy()
    return pred, proba, counts, score_sums


def metric_row(
    *,
    dataset: str,
    seed: int,
    method: str,
    metric: str,
    k: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    total_available_mean: float,
    retained_mean: float,
    repeat: typing.Optional[int] = None,
) -> dict:
    n_classes = y_score.shape[1]
    labels = list(range(n_classes))
    average = "binary" if n_classes == 2 else "macro"
    weighted_f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    primary_f1 = (
        sklearn.metrics.f1_score(y_true, y_pred, average="binary", zero_division=0)
        if n_classes == 2
        else macro_f1
    )
    try:
        auc = (
            sklearn.metrics.roc_auc_score(y_true, y_score[:, 1])
            if n_classes == 2
            else sklearn.metrics.roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=labels)
        )
    except ValueError:
        auc = float("nan")

    confusion = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=labels)
    row = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "metric": metric,
        "k": int(k),
        "repeat": repeat,
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "precision": sklearn.metrics.precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0),
        "recall": sklearn.metrics.recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0),
        "primary_f1": primary_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auc_roc": auc,
        "total_available_mean": total_available_mean,
        "retained_mean": retained_mean,
        "compression_ratio": retained_mean / total_available_mean if total_available_mean else float("nan"),
        "confusion_matrix": json.dumps(confusion.tolist()),
    }
    if n_classes == 2:
        tn, fp, fn, tp = confusion.ravel()
        row.update({"true_positive": int(tp), "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn)})
    return row


def enabled_work(run_dir: Path, dataset: str, seed: int, methods: list[str], metrics: list[str], force: bool) -> dict:
    work = {"vanilla": False, "global_metrics": [], "random": False}
    if "vanilla_lazyfca" in methods:
        path = chunk_path(run_dir, dataset, seed, "vanilla_lazyfca", VANILLA_METRIC)
        if force or not path.exists():
            work["vanilla"] = True
        else:
            append_manifest(
                run_dir,
                {"type": "chunk", "status": "skipped_existing", "dataset": dataset, "seed": seed, "method": "vanilla_lazyfca", "metric": VANILLA_METRIC},
            )

    if "global_topk" in methods:
        for metric in metrics:
            path = chunk_path(run_dir, dataset, seed, "global_topk", metric)
            if force or not path.exists():
                work["global_metrics"].append(metric)
            else:
                append_manifest(
                    run_dir,
                    {"type": "chunk", "status": "skipped_existing", "dataset": dataset, "seed": seed, "method": "global_topk", "metric": metric},
                )

    if "random_topk" in methods:
        path = chunk_path(run_dir, dataset, seed, "random_topk", RANDOM_METRIC)
        if force or not path.exists():
            work["random"] = True
        else:
            append_manifest(
                run_dir,
                {"type": "chunk", "status": "skipped_existing", "dataset": dataset, "seed": seed, "method": "random_topk", "metric": RANDOM_METRIC},
            )
    return work


def no_work(work: dict) -> bool:
    return not work["vanilla"] and not work["global_metrics"] and not work["random"]


def make_prediction_store(
    work: dict,
    k_values: list[int],
    random_repeats: int,
    full_k: int,
) -> dict[tuple[str, str, int, typing.Optional[int]], dict[str, list]]:
    store = {}
    if work["vanilla"]:
        store[("vanilla_lazyfca", VANILLA_METRIC, full_k, None)] = {"pred": [], "score": [], "retained": [], "available": []}
    for metric in work["global_metrics"]:
        for k in k_values:
            store[("global_topk", metric, k, None)] = {"pred": [], "score": [], "retained": [], "available": []}
    if work["random"]:
        for repeat in range(random_repeats):
            for k in k_values:
                store[("random_topk", RANDOM_METRIC, k, repeat)] = {"pred": [], "score": [], "retained": [], "available": []}
    return store


def add_prediction(
    bucket: dict[str, list],
    pred: int,
    proba: np.ndarray,
    retained: int,
    available: int,
) -> None:
    bucket["pred"].append(pred)
    bucket["score"].append(proba)
    bucket["retained"].append(retained)
    bucket["available"].append(available)


def update_diagnostic_counts(explanation, diagnostic: dict) -> None:
    counts = [len(classifiers) for classifiers in explanation.class_classifiers]
    total_available = int(sum(counts))
    diagnostic["total_by_query"].append(total_available)
    for classifiers in explanation.class_classifiers:
        for classifier in classifiers:
            diagnostic["total_classifiers"] += 1
            if classifier.metrics.tp == 1:
                diagnostic["singleton"] += 1
            if classifier.metrics.fp == 0:
                diagnostic["fp_zero"] += 1


def build_streaming_diagnostics(payload: dict, diagnostic: dict) -> dict:
    train_counts = np.bincount(payload["y_train"], minlength=payload["n_classes"])
    test_counts = np.bincount(payload["y_test"], minlength=payload["n_classes"])
    total_by_query = diagnostic["total_by_query"] or [0]
    total = diagnostic["total_classifiers"]
    return {
        "dataset": payload["dataset"],
        "seed": payload["seed"],
        "n_classes": payload["n_classes"],
        "train_size": int(len(payload["y_train"])),
        "test_size": int(len(payload["y_test"])),
        "train_class_counts": json.dumps(train_counts.tolist()),
        "test_class_counts": json.dumps(test_counts.tolist()),
        "numeric_feature_count": payload["numeric_feature_count"],
        "categorical_feature_count": payload["categorical_feature_count"],
        "encoded_feature_count": payload["encoded_feature_count"],
        "total_classifiers_mean": float(np.mean(total_by_query)),
        "total_classifiers_min": int(np.min(total_by_query)),
        "total_classifiers_max": int(np.max(total_by_query)),
        "singleton_tp1_rate": diagnostic["singleton"] / total if total else float("nan"),
        "fp_zero_rate": diagnostic["fp_zero"] / total if total else float("nan"),
    }


def stream_evaluate_chunks(
    run_dir: Path,
    payload: dict,
    work: dict,
    k_values: list[int],
    random_repeats: int,
) -> dict:
    dataset = payload["dataset"]
    seed = int(payload["seed"])
    n_classes = payload["n_classes"]
    priors = payload["class_priors"]
    full_k = len(payload["y_train"])
    store = make_prediction_store(work, k_values, random_repeats, full_k)
    diagnostic = {"total_by_query": [], "total_classifiers": 0, "singleton": 0, "fp_zero": 0}

    for query_idx, (_row_idx, sample) in enumerate(payload["X_test"].iterrows()):
        explanation = payload["model"].explain_sample(sample)
        update_diagnostic_counts(explanation, diagnostic)
        available = sum(len(classifiers) for classifiers in explanation.class_classifiers)

        if work["vanilla"]:
            counts = np.asarray([len(classifiers) for classifiers in explanation.class_classifiers], dtype=float)
            pred = choose_class(counts, counts.copy(), priors)
            proba = counts / counts.sum() if counts.sum() > 0 else priors.copy()
            add_prediction(store[("vanilla_lazyfca", VANILLA_METRIC, full_k, None)], pred, proba, int(counts.sum()), available)

        for metric in work["global_metrics"]:
            ranked = ranked_classifiers(explanation, metric)
            for k in k_values:
                kept = ranked[: min(k, len(ranked))]
                pred, proba, _counts, _score_sums = predict_from_retained(kept, n_classes, priors)
                add_prediction(store[("global_topk", metric, k, None)], pred, proba, len(kept), available)

        if work["random"]:
            flat = flatten_classifiers(explanation)
            for repeat in range(random_repeats):
                rng = np.random.default_rng(seed * 1_000_003 + query_idx * 1_009 + repeat)
                order = rng.permutation(len(flat))
                ranked = [(flat[i][0], flat[i][1], flat[i][2], 1.0) for i in order]
                for k in k_values:
                    kept = ranked[: min(k, len(ranked))]
                    pred, proba, _counts, _score_sums = predict_from_retained(kept, n_classes, priors)
                    add_prediction(store[("random_topk", RANDOM_METRIC, k, repeat)], pred, proba, len(kept), available)

    by_chunk: dict[tuple[str, str], list[dict]] = {}
    for (method, metric, k, repeat), bucket in store.items():
        row = metric_row(
            dataset=dataset,
            seed=seed,
            method=method,
            metric=metric,
            k=k,
            repeat=repeat,
            y_true=payload["y_test"],
            y_pred=np.asarray(bucket["pred"], dtype=int),
            y_score=np.vstack(bucket["score"]),
            total_available_mean=float(np.mean(bucket["available"])),
            retained_mean=float(np.mean(bucket["retained"])),
        )
        by_chunk.setdefault((method, metric), []).append(row)

    for (method, metric), rows in by_chunk.items():
        path = chunk_path(run_dir, dataset, seed, method, metric)
        df = pd.DataFrame(rows).sort_values(["repeat", "k"], na_position="first")
        write_chunk(path, df)
        append_manifest(
            run_dir,
            {
                "type": "chunk",
                "status": "completed",
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "metric": metric,
                "rows": int(len(df)),
                "path": str(path.relative_to(run_dir)),
            },
        )

    return build_streaming_diagnostics(payload, diagnostic)


def write_chunk(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_diagnostic(run_dir: Path, diagnostic: dict) -> None:
    path = diagnostic_path(run_dir, str(diagnostic["dataset"]), int(diagnostic["seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame([diagnostic]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def combine_chunks(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((run_dir / "chunks").glob("*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def combine_diagnostics(run_dir: Path) -> pd.DataFrame:
    frames = []
    diagnostics_path = run_dir / "dataset_diagnostics.csv"
    if diagnostics_path.exists():
        frames.append(pd.read_csv(diagnostics_path))
    for path in sorted((run_dir / "diagnostics").glob("*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["dataset", "seed"], keep="last")
        .sort_values(["dataset", "seed"])
    )


def summarize_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    vanilla = results[results["method"] == "vanilla_lazyfca"].copy()
    topk = results[results["method"] != "vanilla_lazyfca"].copy()
    group_cols = ["dataset", "method", "metric", "k"]
    summary = (
        topk.groupby(group_cols, dropna=False)
        .agg(
            primary_f1_mean=(PRIMARY_METRIC, "mean"),
            primary_f1_std=(PRIMARY_METRIC, "std"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            weighted_f1_mean=("weighted_f1", "mean"),
            retained_mean=("retained_mean", "mean"),
            compression_ratio_mean=("compression_ratio", "mean"),
            runs=("primary_f1", "count"),
        )
        .reset_index()
    )

    compact_rows = []
    for (dataset, method, metric), df in summary.groupby(["dataset", "method", "metric"], dropna=False):
        df = df.sort_values("k")
        best_idx = df["primary_f1_mean"].idxmax()
        best = df.loc[best_idx]
        row = {
            "dataset": dataset,
            "method": method,
            "metric": metric,
            "best_k": int(best["k"]),
            "best_primary_f1_mean": float(best["primary_f1_mean"]),
            "best_retained_mean": float(best["retained_mean"]),
            "best_compression_ratio_mean": float(best["compression_ratio_mean"]),
        }
        for pct in [1, 3, 5]:
            threshold = row["best_primary_f1_mean"] * (1.0 - pct / 100.0)
            eligible = df[df["primary_f1_mean"] >= threshold].sort_values("k")
            if eligible.empty:
                row[f"smallest_k_within_{pct}pct"] = np.nan
                row[f"primary_f1_within_{pct}pct"] = np.nan
            else:
                selected = eligible.iloc[0]
                row[f"smallest_k_within_{pct}pct"] = int(selected["k"])
                row[f"primary_f1_within_{pct}pct"] = float(selected["primary_f1_mean"])
        compact_rows.append(row)
    compact = pd.DataFrame(compact_rows).sort_values(["dataset", "best_primary_f1_mean"], ascending=[True, False])
    return vanilla, summary, compact


def write_plots(run_dir: Path, summary: pd.DataFrame, vanilla: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot_dir = run_dir / "plots"
    metric_plot_dir = plot_dir / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_plot_dir.mkdir(parents=True, exist_ok=True)
    for dataset, df in summary.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for (method, metric), sub in df.groupby(["method", "metric"], dropna=False):
            sub = sub.sort_values("k")
            label = f"{method}:{metric}"
            ax.plot(sub["k"], sub["primary_f1_mean"], marker="o", linewidth=1.5, markersize=3, label=label)
        vanilla_sub = vanilla[vanilla["dataset"] == dataset]
        if not vanilla_sub.empty:
            ax.axhline(
                vanilla_sub[PRIMARY_METRIC].mean(),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="vanilla LazyFCA",
            )
        ax.set_xscale("log")
        ax.set_xlabel("k retained classifiers")
        ax.set_ylabel("primary F1")
        ax.set_title(f"{dataset}: compactness-first top-k ranking")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_topk_primary_f1.png", dpi=180)
        plt.close(fig)

        vanilla_sub = vanilla[vanilla["dataset"] == dataset]
        vanilla_f1 = None if vanilla_sub.empty else float(vanilla_sub[PRIMARY_METRIC].mean())
        for (method, metric), sub in df.groupby(["method", "metric"], dropna=False):
            sub = sub.sort_values("k")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(
                sub["k"],
                sub["primary_f1_mean"],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=f"{method}:{metric}",
            )
            if "primary_f1_std" in sub and sub["primary_f1_std"].notna().any():
                lower = sub["primary_f1_mean"] - sub["primary_f1_std"].fillna(0.0)
                upper = sub["primary_f1_mean"] + sub["primary_f1_std"].fillna(0.0)
                ax.fill_between(sub["k"], lower, upper, alpha=0.15)
            if vanilla_f1 is not None:
                ax.axhline(
                    vanilla_f1,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="vanilla LazyFCA",
                )
            ax.set_xscale("log")
            ax.set_xlabel("k retained classifiers")
            ax.set_ylabel("primary F1")
            ax.set_title(f"{dataset}: {method}:{metric}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(metric_plot_dir / f"{dataset}__{method}__{metric}_topk_primary_f1.png", dpi=180)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run incremental paper experiments for ranked LazyFCA.")
    parser.add_argument("--config", default="paper_experiments/config.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    config = load_config(config_path)
    if config.get("cache_explanations", False):
        raise ValueError(
            "cache_explanations=true is disabled because full LazyFCA explanations can require excessive RAM. "
            "Use the default streaming evaluator instead."
        )
    if args.smoke:
        config = {**config, "seeds": [0], "run_name": "smoke"}

    run_name = args.run_name or config.get("run_name", "default")
    output_dir = ROOT / config.get("output_dir", "paper_experiments/results")
    run_dir = output_dir / run_name
    for subdir in ["chunks", "diagnostics", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset_names = selected_names(enabled_names(config["datasets"]), args.datasets)
    metric_names = selected_names(enabled_names(config["metrics"]), args.metrics)
    unknown_metrics = sorted(set(metric_names).difference(METADATA_DICT))
    if unknown_metrics:
        raise ValueError(f"Unknown LazyFCA metric names in config/CLI: {unknown_metrics}")
    method_names = selected_names(enabled_names(config["methods"]), args.methods)
    seeds = args.seeds if args.seeds is not None else list(config.get("seeds", [0, 1, 2, 3, 4]))
    force = bool(args.force or config.get("force", False))
    random_repeats = int(config.get("random_topk_repeats", 5))

    if args.smoke:
        dataset_names = dataset_names[:1]
        metric_names = metric_names[: min(2, len(metric_names))]
        random_repeats = 1

    config_fingerprint = short_hash(
        {
            "config_path": str(config_path.relative_to(ROOT)),
            "datasets": dataset_names,
            "metrics": metric_names,
            "methods": method_names,
            "seeds": seeds,
            "test_size": config.get("test_size"),
            "k_grid": config.get("k_grid"),
            "random_topk_repeats": random_repeats,
        }
    )
    append_manifest(
        run_dir,
        {
            "type": "run",
            "status": "started",
            "config_fingerprint": config_fingerprint,
            "datasets": dataset_names,
            "metrics": metric_names,
            "methods": method_names,
            "seeds": seeds,
            "force": force,
            "smoke": bool(args.smoke),
        },
    )

    diagnostics = []
    for spec in dataset_specs(config, dataset_names):
        for seed in seeds:
            work = enabled_work(run_dir, spec.name, seed, method_names, metric_names, force=force)
            payload = prepare_dataset(config, spec, seed)
            full_k = len(payload["y_train"])
            k_values = k_values_from_config(config.get("k_grid", {}), full_k=full_k, smoke=bool(args.smoke))
            diagnostics_missing = force or not diagnostic_path(run_dir, spec.name, seed).exists()
            if no_work(work) and not diagnostics_missing:
                print(f"[skip]  {spec.name} seed={seed}: all chunks already exist", flush=True)
                continue

            started = time.time()
            mode = "diagnostic" if no_work(work) else "stream"
            print(
                f"[{mode}] {spec.name} seed={seed} "
                f"metrics={len(work['global_metrics'])} k={len(k_values)}",
                flush=True,
            )
            diagnostic = stream_evaluate_chunks(
                run_dir,
                payload,
                work=work,
                k_values=k_values,
                random_repeats=random_repeats,
            )
            write_diagnostic(run_dir, diagnostic)
            diagnostics.append(diagnostic)
            append_manifest(
                run_dir,
                {
                    "type": "dataset_seed",
                    "status": "completed",
                    "dataset": spec.name,
                    "seed": seed,
                    "seconds": round(time.time() - started, 3),
                    "streaming": True,
                    "diagnostics_only": bool(no_work(work)),
                },
            )

    diagnostics_df = combine_diagnostics(run_dir)
    if not diagnostics_df.empty:
        diagnostics_df.to_csv(run_dir / "dataset_diagnostics.csv", index=False)

    results = combine_chunks(run_dir)
    if not results.empty:
        results.to_csv(run_dir / "topk_results.csv", index=False)
        vanilla, summary, compact = summarize_results(results)
        vanilla.to_csv(run_dir / "vanilla_lazyfca.csv", index=False)
        summary.to_csv(run_dir / "summary_by_dataset_metric.csv", index=False)
        plot_columns = [
            "dataset",
            "method",
            "metric",
            "k",
            "primary_f1_mean",
            "primary_f1_std",
            "accuracy_mean",
            "macro_f1_mean",
            "weighted_f1_mean",
            "retained_mean",
            "compression_ratio_mean",
            "runs",
        ]
        summary[[col for col in plot_columns if col in summary.columns]].to_csv(
            run_dir / "topk_plot_data.csv",
            index=False,
        )
        compact.to_csv(run_dir / "compactness_summary.csv", index=False)
        write_plots(run_dir, summary, vanilla)

    append_manifest(run_dir, {"type": "run", "status": "completed", "config_fingerprint": config_fingerprint})
    print(f"Done. Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
