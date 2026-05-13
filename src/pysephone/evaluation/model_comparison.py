"""
Statistical comparison of phenology model performance across multiple
runs and datasets.

Given a collection of completed evaluations — each tagged with its
``(model_key, dataset_key, seed)`` identity — this module:

  1. Builds a scores matrix (datasets x models) by aggregating a metric
     across seeds. The handling of missing entries is selected via the
     ``missing_policy`` argument (see :class:`MissingPolicy`).
  2. Runs the Friedman omnibus test and Nemenyi post-hoc.
  3. Optionally produces an ``autorank`` report + critical-difference plot,
     plus simple comparison plots (rank distribution, Nemenyi heatmap,
     scores heatmap) — each in its own figure.

The module deliberately does NOT load evaluations from disk or parse CLI
arguments. Callers pass already-loaded
:class:`~pysephone.evaluation.regression.SingleTargetRegression` instances.

Persistence
-----------
Calling :meth:`ComparisonReport.save` writes the report under
``<data_root>/outputs/comparisons/<comparison_id>/``:

    ├── scores.csv         # datasets x models matrix
    ├── nemenyi.csv        # pairwise post-hoc p-values
    ├── summary.txt        # textual summary
    ├── metadata.json      # metric, alpha, statistic, p-value, etc.
    └── plots/             # optional, only if save_plots=True
        ├── scores_heatmap.png
        ├── nemenyi_heatmap.png
        ├── rank_distribution.png
        └── critical_difference.png

``<data_root>`` is :func:`pysephone.paths.get_data_root` (overridable via the
``PYSEPHONE_DATA_ROOT`` env var), matching where ``SingleTargetRegression``
already saves individual evaluations.

Optional dependencies
---------------------
``scikit-posthocs`` and ``autorank`` are not hard requirements; install via
the ``stats`` extra::

    pip install pysephone[stats]

Example
-------
::

    runs = []
    for model_key, dataset_key, seed in product(model_keys, dataset_keys, seeds):
        eval_result = SingleTargetRegression.load(
            f"{run_prefix}_{model_key}_{dataset_key}_seed_{seed}"
        )
        runs.append(EvaluationRun(eval_result, model_key, dataset_key, seed))

    report = compare_models(
        runs,
        model_keys=model_keys,
        dataset_keys=dataset_keys,
        display_names=model_display_names,
        metric="mae",
        alpha=0.05,
        missing_policy="skip_seed",
    )
    print(report.summary())
    report.save("exp5_split_ts_mae", save_plots=True)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pysephone.evaluation.regression import SingleTargetRegression
from pysephone.paths import get_comparisons_dir, get_data_root


# ---------------------------------------------------------------------------
# Style (mirrors regression.py)
# ---------------------------------------------------------------------------

_STYLE = {
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
}


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

class MissingPolicy(str, Enum):
    """How to react when a ``(model, dataset, seed)`` run is absent.

    Values:
        SKIP_SEED:    Drop the ``(dataset, seed)`` combination across all
                      models; other seeds for the same dataset still
                      contribute. Matches the typical "drop the row, keep the
                      study" behaviour.
        SKIP_DATASET: If any expected run is missing for a dataset, drop the
                      entire dataset row from the comparison.
        RAISE:        Raise :class:`RuntimeError` on the first missing run.
    """
    SKIP_SEED = "skip_seed"
    SKIP_DATASET = "skip_dataset"
    RAISE = "raise"

    @classmethod
    def from_key(cls, key: 'str | MissingPolicy') -> 'MissingPolicy':
        if isinstance(key, cls):
            return key
        try:
            return cls(key)
        except ValueError as e:
            valid = [m.value for m in cls]
            raise ValueError(
                f"Unknown missing_policy {key!r}. Valid keys: {valid}"
            ) from e


@dataclass(frozen=True)
class EvaluationRun:
    """One completed evaluation tagged with its model/dataset/seed identity.

    Attributes:
        eval_result: A completed :class:`SingleTargetRegression`.
        model_key:   Canonical model identifier (e.g. dotted import path).
        dataset_key: Canonical dataset identifier.
        seed:        Random seed used for the run, or ``None`` for
                     deterministic runs.
    """
    eval_result: SingleTargetRegression
    model_key: str
    dataset_key: str
    seed: Optional[int] = None


@dataclass
class ComparisonReport:
    """Output of :func:`compare_models`.

    Attributes:
        scores:               Datasets x models matrix of aggregated metric
                              values.
        statistic:            Friedman chi-square statistic.
        p_value:              Friedman p-value.
        nemenyi:              Pairwise Nemenyi p-value matrix.
        best_model:           Best model name under ``order``.
        is_best_significant:  ``True`` iff ``best_model`` is significantly
                              better than every other model at ``alpha``.
        alpha:                Significance threshold.
        order:                ``"ascending"`` (lower is better) or
                              ``"descending"`` (higher is better).
        metric:               Aggregated metric name.
        n_skipped_seeds:      Count of ``(dataset, seed)`` combinations
                              dropped due to missing runs.
        n_skipped_datasets:   Count of datasets dropped under
                              ``missing_policy="skip_dataset"``.
        missing_policy:       The policy that produced this report.
    """
    scores: pd.DataFrame
    statistic: float
    p_value: float
    nemenyi: pd.DataFrame
    best_model: str
    is_best_significant: bool
    alpha: float
    order: str
    metric: str
    n_skipped_seeds: int = 0
    n_skipped_datasets: int = 0
    missing_policy: str = MissingPolicy.SKIP_SEED.value

    def summary(self) -> str:
        means = self.scores.mean(axis=0).sort_values(
            ascending=(self.order == "ascending")
        )
        ranking = "\n".join(
            f"  {i + 1}. {name}: {value:.4f}"
            for i, (name, value) in enumerate(means.items())
        )
        return (
            f"Friedman chi-square = {self.statistic:.4f}  "
            f"(p = {self.p_value:.4g})\n"
            f"Best model ({self.metric}, {self.order}): {self.best_model}\n"
            f"Significantly better than all others at alpha={self.alpha}: "
            f"{self.is_best_significant}\n"
            f"Missing-entry policy: {self.missing_policy}  "
            f"(skipped seeds={self.n_skipped_seeds}, "
            f"skipped datasets={self.n_skipped_datasets})\n"
            f"Mean {self.metric} per model:\n{ranking}"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        comparison_id: str,
        *,
        root: Optional[Path] = None,
        save_plots: bool = False,
        plot_dpi: int = 200,
    ) -> Path:
        """Persist scores, nemenyi matrix, summary, and metadata.

        Args:
            comparison_id: Folder name under ``outputs/comparisons/``.
            root:          Data root. Defaults to
                           :func:`pysephone.paths.get_data_root`.
            save_plots:    If ``True``, also render and save all plots into
                           a ``plots/`` subdirectory. ``critical_difference``
                           is skipped silently if ``autorank`` is unavailable.
            plot_dpi:      Resolution for saved plots.

        Returns:
            The output directory path.
        """
        d = get_comparisons_dir(root or get_data_root()) / comparison_id
        d.mkdir(parents=True, exist_ok=True)
        self.scores.to_csv(d / "scores.csv")
        self.nemenyi.to_csv(d / "nemenyi.csv")
        (d / "summary.txt").write_text(self.summary())
        metadata = {
            "metric": self.metric,
            "alpha": self.alpha,
            "order": self.order,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "best_model": self.best_model,
            "is_best_significant": self.is_best_significant,
            "missing_policy": self.missing_policy,
            "n_skipped_seeds": self.n_skipped_seeds,
            "n_skipped_datasets": self.n_skipped_datasets,
            "n_datasets": int(self.scores.shape[0]),
            "n_models": int(self.scores.shape[1]),
            "models": list(self.scores.columns),
            "datasets": list(self.scores.index),
        }
        with open(d / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if save_plots:
            plots_dir = d / "plots"
            plots_dir.mkdir(exist_ok=True)
            for name, fig in [
                ("scores_heatmap", plot_score_heatmap(self.scores, metric=self.metric)),
                ("nemenyi_heatmap", plot_nemenyi_heatmap(self)),
                ("rank_distribution", plot_rank_distribution(self.scores, order=self.order)),
            ]:
                fig.savefig(plots_dir / f"{name}.png", dpi=plot_dpi, bbox_inches="tight")
                plt.close(fig)
            try:
                ar = autorank_report(self.scores, alpha=self.alpha, order=self.order)
                fig = plot_critical_difference(ar)
                fig.savefig(plots_dir / "critical_difference.png", dpi=plot_dpi, bbox_inches="tight")
                plt.close(fig)
            except ImportError:
                pass

        return d


# ---------------------------------------------------------------------------
# Building the scores table
# ---------------------------------------------------------------------------

def _extract_metric(
    eval_result: SingleTargetRegression,
    metric: str,
    split: str,
) -> float:
    metrics = eval_result.compute_metrics()
    if split not in metrics:
        raise KeyError(f"Split {split!r} not found in compute_metrics() output")
    split_metrics = metrics[split]
    if metric not in split_metrics:
        raise KeyError(
            f"Metric {metric!r} not in split {split!r}. "
            f"Available: {sorted(split_metrics)}"
        )
    return float(split_metrics[metric])


def build_scores_table(
    runs: Iterable[EvaluationRun],
    *,
    model_keys: Sequence[str],
    dataset_keys: Sequence[str],
    seeds: Optional[Sequence[Optional[int]]] = None,
    metric: str = "mae",
    split: str = "test",
    display_names: Optional[Mapping[str, str]] = None,
    missing_policy: 'str | MissingPolicy' = MissingPolicy.SKIP_SEED,
    seed_reduce: str = "mean",
) -> Tuple[pd.DataFrame, int, int]:
    """Aggregate runs into a (datasets x models) scores matrix.

    For each ``(dataset, seed)`` combination, every model in ``model_keys``
    must have a corresponding run; otherwise ``missing_policy`` decides what
    happens. Surviving seeds are then reduced (mean/median) into a single
    value per ``(dataset, model)``.

    Args:
        runs:           Iterable of :class:`EvaluationRun`.
        model_keys:     Models to compare, in column order.
        dataset_keys:   Datasets to include, in row order.
        seeds:          Seeds to require. If ``None``, inferred from ``runs``.
        metric:         Metric key inside ``compute_metrics()[split]``.
        split:          ``"train"`` or ``"test"``.
        display_names:  Optional ``model_key -> pretty name`` map for columns.
        missing_policy: See :class:`MissingPolicy`. Accepts the enum value or
                        its string key (``"skip_seed"``, ``"skip_dataset"``,
                        ``"raise"``).
        seed_reduce:    ``"mean"`` or ``"median"``.

    Returns:
        Tuple ``(scores_df, n_skipped_seeds, n_skipped_datasets)``.
    """
    if seed_reduce not in {"mean", "median"}:
        raise ValueError(f"seed_reduce must be 'mean' or 'median', got {seed_reduce!r}")
    policy = MissingPolicy.from_key(missing_policy)

    by_key: dict[Tuple[str, str, Optional[int]], EvaluationRun] = {}
    for r in runs:
        by_key[(r.model_key, r.dataset_key, r.seed)] = r

    if seeds is None:
        observed_seeds = {r.seed for r in by_key.values()}
        seeds = sorted(s for s in observed_seeds if s is not None) or [None]

    skipped_seeds = 0
    skipped_datasets = 0
    per_dataset: dict[str, list[list[float]]] = {dk: [] for dk in dataset_keys}

    for dk in dataset_keys:
        dataset_seeds: list[list[float]] = []
        dataset_complete = True
        for s in seeds:
            row: list[float] = []
            complete = True
            for mk in model_keys:
                run = by_key.get((mk, dk, s))
                if run is None:
                    complete = False
                    break
                try:
                    row.append(_extract_metric(run.eval_result, metric, split))
                except KeyError:
                    complete = False
                    break
            if complete:
                dataset_seeds.append(row)
            else:
                if policy is MissingPolicy.RAISE:
                    raise RuntimeError(
                        f"Missing run(s) for dataset={dk!r}, seed={s!r}; "
                        f"at least one model in model_keys has no EvaluationRun."
                    )
                dataset_complete = False
                skipped_seeds += 1

        if policy is MissingPolicy.SKIP_DATASET and not dataset_complete:
            skipped_datasets += 1
            continue
        per_dataset[dk] = dataset_seeds

    reducer = np.mean if seed_reduce == "mean" else np.median
    rows: list[np.ndarray] = []
    index: list[str] = []
    for dk in dataset_keys:
        seed_rows = per_dataset.get(dk, [])
        if not seed_rows:
            continue
        rows.append(reducer(seed_rows, axis=0))
        index.append(dk)

    if not rows:
        raise RuntimeError(
            "No (dataset, seed) combinations had complete coverage across all "
            "model_keys. Either runs are missing or the keys do not match."
        )

    cols = (
        [display_names.get(mk, mk) for mk in model_keys]
        if display_names else list(model_keys)
    )
    if len(set(cols)) != len(cols):
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        raise ValueError(f"Duplicate model column names: {dupes}")

    return pd.DataFrame(rows, columns=cols, index=index), skipped_seeds, skipped_datasets


# ---------------------------------------------------------------------------
# Friedman + Nemenyi
# ---------------------------------------------------------------------------

def friedman_nemenyi(
    scores: pd.DataFrame,
    *,
    alpha: float = 0.05,
    order: str = "ascending",
) -> ComparisonReport:
    """Friedman omnibus + Nemenyi post-hoc on a scores matrix.

    Args:
        scores: Datasets x models matrix (as produced by
                :func:`build_scores_table`).
        alpha:  Significance threshold.
        order:  ``"ascending"`` (lower is better, e.g. MAE/RMSE) or
                ``"descending"`` (higher is better, e.g. R^2).

    Returns:
        :class:`ComparisonReport` with ``metric`` / ``n_skipped_*`` left at
        defaults — prefer :func:`compare_models` for full population.
    """
    if order not in {"ascending", "descending"}:
        raise ValueError(f"order must be 'ascending' or 'descending', got {order!r}")
    if scores.shape[1] < 2:
        raise ValueError("Need at least 2 models to compare")
    if scores.shape[0] < 2:
        raise ValueError("Need at least 2 datasets for the Friedman test")

    from scipy.stats import friedmanchisquare

    try:
        import scikit_posthocs as sp
    except ImportError as e:
        raise ImportError(
            "scikit-posthocs is required for the Nemenyi post-hoc test. "
            "Install with: pip install pysephone[stats]"
        ) from e

    arr = scores.to_numpy()
    stat, p_value = friedmanchisquare(*[arr[:, i] for i in range(arr.shape[1])])

    nemenyi = sp.posthoc_nemenyi_friedman(arr)
    nemenyi.index = scores.columns
    nemenyi.columns = scores.columns

    means = scores.mean(axis=0)
    best = means.idxmin() if order == "ascending" else means.idxmax()
    is_best = all(
        nemenyi.loc[best, other] < alpha
        for other in nemenyi.columns
        if other != best
    )

    return ComparisonReport(
        scores=scores,
        statistic=float(stat),
        p_value=float(p_value),
        nemenyi=nemenyi,
        best_model=str(best),
        is_best_significant=bool(is_best),
        alpha=alpha,
        order=order,
        metric="",
    )


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------

def compare_models(
    runs: Iterable[EvaluationRun],
    *,
    model_keys: Sequence[str],
    dataset_keys: Sequence[str],
    seeds: Optional[Sequence[Optional[int]]] = None,
    metric: str = "mae",
    split: str = "test",
    alpha: float = 0.05,
    order: str = "ascending",
    display_names: Optional[Mapping[str, str]] = None,
    missing_policy: 'str | MissingPolicy' = MissingPolicy.SKIP_SEED,
    seed_reduce: str = "mean",
) -> ComparisonReport:
    """Build the scores table and run Friedman + Nemenyi in one shot.

    See :func:`build_scores_table` and :func:`friedman_nemenyi` for argument
    semantics, and :class:`MissingPolicy` for missing-entry handling.
    """
    scores, skipped_seeds, skipped_datasets = build_scores_table(
        runs,
        model_keys=model_keys,
        dataset_keys=dataset_keys,
        seeds=seeds,
        metric=metric,
        split=split,
        display_names=display_names,
        missing_policy=missing_policy,
        seed_reduce=seed_reduce,
    )
    report = friedman_nemenyi(scores, alpha=alpha, order=order)
    report.metric = metric
    report.n_skipped_seeds = skipped_seeds
    report.n_skipped_datasets = skipped_datasets
    report.missing_policy = MissingPolicy.from_key(missing_policy).value
    return report


# ---------------------------------------------------------------------------
# Plots (each returns its own Figure)
# ---------------------------------------------------------------------------

def plot_score_heatmap(
    scores: pd.DataFrame,
    *,
    metric: str = "",
    cmap: str = "viridis_r",
    annotate: bool = True,
) -> Figure:
    """Heatmap of the aggregated scores matrix (datasets x models).

    Returns its own :class:`Figure` (separate window).
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(max(5.0, 0.7 * scores.shape[1] + 2.5),
                     max(3.0, 0.5 * scores.shape[0] + 1.5)),
        )
        ax.grid(False)
        im = ax.imshow(scores.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(scores.shape[1]))
        ax.set_xticklabels(scores.columns, rotation=30, ha="right")
        ax.set_yticks(range(scores.shape[0]))
        ax.set_yticklabels(scores.index)
        ax.set_title(f"Mean {metric} per (dataset, model)" if metric
                     else "Mean score per (dataset, model)")

        if annotate:
            vals = scores.values
            vmid = np.nanmean(vals)
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    if np.isnan(v):
                        continue
                    colour = "white" if v > vmid else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color=colour)

        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(metric or "score")
        fig.tight_layout()
    return fig


def plot_nemenyi_heatmap(
    report: ComparisonReport,
    *,
    cmap: str = "RdYlGn",
) -> Figure:
    """Heatmap of pairwise Nemenyi post-hoc p-values.

    Cells below ``report.alpha`` are annotated with ``*``.
    """
    nem = report.nemenyi
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(max(5.0, 0.7 * nem.shape[1] + 2.5),
                     max(4.0, 0.7 * nem.shape[0] + 1.5)),
        )
        ax.grid(False)
        im = ax.imshow(nem.values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(nem.shape[1]))
        ax.set_xticklabels(nem.columns, rotation=30, ha="right")
        ax.set_yticks(range(nem.shape[0]))
        ax.set_yticklabels(nem.index)
        ax.set_title(f"Nemenyi pairwise p-values (alpha={report.alpha})")

        vals = nem.values
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if i == j:
                    continue
                v = vals[i, j]
                if np.isnan(v):
                    continue
                marker = "*" if v < report.alpha else ""
                ax.text(j, i, f"{v:.2f}{marker}", ha="center", va="center",
                        fontsize=8,
                        color=("black" if v > 0.5 else "white"))

        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("p-value")
        fig.tight_layout()
    return fig


def plot_rank_distribution(
    scores: pd.DataFrame,
    *,
    order: str = "ascending",
) -> Figure:
    """Boxplot of per-dataset ranks for each model, with mean rank overlaid.

    Ranks are computed per dataset row — rank 1 is the best under ``order``.
    """
    ascending = order == "ascending"
    ranks = scores.rank(axis=1, ascending=ascending, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values(ascending=True)
    ordered_models = list(mean_ranks.index)
    data = [ranks[m].values for m in ordered_models]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(max(5.0, 0.7 * len(ordered_models) + 2.0), 4.5)
        )
        ax.boxplot(
            data, vert=True, patch_artist=True, widths=0.55,
            medianprops=dict(color="#333333", linewidth=1.4),
            boxprops=dict(facecolor="#dbe8f4", edgecolor="#4878d0"),
            whiskerprops=dict(color="#4878d0"),
            capprops=dict(color="#4878d0"),
            flierprops=dict(marker="o", markersize=4, alpha=0.6),
        )
        for i, m in enumerate(ordered_models, start=1):
            ax.scatter([i], [mean_ranks[m]], color="#ee854a", zorder=3,
                       label="Mean rank" if i == 1 else None, s=40, marker="D")

        ax.set_xticks(range(1, len(ordered_models) + 1))
        ax.set_xticklabels(ordered_models, rotation=30, ha="right")
        ax.invert_yaxis()  # rank 1 at the top
        ax.set_ylabel("Rank (1 = best)")
        ax.set_title(f"Per-dataset rank distribution ({order})")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# autorank wrappers (optional dependency)
# ---------------------------------------------------------------------------

def autorank_report(
    scores: pd.DataFrame,
    *,
    alpha: float = 0.05,
    order: str = "ascending",
    force_mode: Optional[str] = "nonparametric",
    verbose: bool = False,
) -> Any:
    """Run ``autorank.autorank`` on a scores matrix.

    Lazy import; install with ``pip install pysephone[stats]``.
    """
    try:
        from autorank import autorank
    except ImportError as e:
        raise ImportError(
            "autorank is required. Install with: pip install pysephone[stats]"
        ) from e
    kwargs: dict[str, Any] = dict(alpha=alpha, order=order, verbose=verbose)
    if force_mode is not None:
        kwargs["force_mode"] = force_mode
    return autorank(scores, **kwargs)


def plot_critical_difference(
    autorank_result: Any,
    *,
    savefig: Optional[str] = None,
    dpi: int = 200,
) -> Figure:
    """Render a critical-difference diagram from an ``autorank`` result.

    Args:
        autorank_result: Value returned by :func:`autorank_report`.
        savefig:         Optional output path; if provided the figure is saved
                         with ``bbox_inches="tight"`` at the given ``dpi``.

    Returns:
        The current :class:`matplotlib.figure.Figure`.
    """
    try:
        from autorank import plot_stats
    except ImportError as e:
        raise ImportError(
            "autorank is required. Install with: pip install pysephone[stats]"
        ) from e

    plot_stats(autorank_result)
    fig = plt.gcf()
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
    return fig
