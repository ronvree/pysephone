"""
Generate visually rich README figures and save them to outputs/readme/.

All figures are self-contained — they use synthetic temperature series and
literature-based model parameters, so no external data files are required.

Figures produced:

1. ``season_dynamics``     — Temperature bars coloured by chilling (blue) vs
                              forcing (orange) contribution, with cumulative
                              chill and forcing curves for all CF models shown
                              for two contrasting winter scenarios.

2. ``chill_response``      — Temperature response functions: Utah, ChillingDays,
                              Dynamic chill models (left) and GDD forcing (right).

3. ``winter_sensitivity``  — Bloom DOY vs mean winter temperature for each model,
                              showing the chilling-gate failure cliff and the
                              spread across noise realisations.

Usage::

    python -m pysephone.run.readme_figures [--dpi 180] [--formats png svg]
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pysephone.models.util.func_phenology import (
    func_chilling_days,
    func_dynamic_chill_daily,
    func_utah_chill,
)
from pysephone.paths import get_repo_root
from pysephone.visualize.dataset import save_figure

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared palette & model colours  (mirrors model_exploration.ipynb)
# ---------------------------------------------------------------------------

PALETTE = {
    'temp_warm':  '#f4a46280',
    'temp_cold':  '#7eb8d480',
    'temp_line':  '#c0392b',
    'chill':      '#2980b9',
    'force':      '#d35400',
    'dechill':    '#c0392b',
    'none':       '#d5d5d5',
    'zero':       '#cccccc',
    'thresh':     '#555555',
    'bg':         '#f9f9f9',
}

MODEL_COLORS = {
    'GDD':              '#888888',
    'Utah + GDD':       '#2980b9',
    'ChillingDays + GDD': '#27ae60',
    'Dynamic + GDD':    '#8e44ad',
}

# Month ticks for a season starting Oct 1  (day 0 = Oct 1)
_MONTH_STARTS  = [0,  31,  61,  92, 122, 153, 181, 212, 243, 273, 304, 334]
_MONTH_LABELS  = ['Oct','Nov','Dec','Jan','Feb','Mar',
                  'Apr','May','Jun','Jul','Aug','Sep']

def _month_ticks(n_days: int):
    ticks  = [d for d in _MONTH_STARTS if d < n_days]
    labels = [_MONTH_LABELS[i] for i, d in enumerate(_MONTH_STARTS) if d < n_days]
    return ticks, labels

def _style_time_ax(ax, n_days: int, ylabel: str = '', xlabel: bool = False,
                   ylim=None, hide_x: bool = False) -> None:
    ax.set_xlim(0, n_days - 1)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ticks, labels = _month_ticks(n_days)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if (xlabel and not hide_x) else [], fontsize=8)
    if xlabel and not hide_x:
        ax.set_xlabel('Month  (season starting Oct 1)', fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)


# ---------------------------------------------------------------------------
# Synthetic temperature generation  (from model_exploration.ipynb)
# ---------------------------------------------------------------------------

def _generate_season(
    mean_winter: float = 4.0,
    mean_summer: float = 22.0,
    noise_std: float = 2.5,
    n_days: int = 270,
    seed: int = 42,
) -> np.ndarray:
    """Cosine seasonal cycle with Gaussian noise, starting Oct 1."""
    rng = np.random.default_rng(seed)
    mid_winter = 92          # ~Jan 1
    days = np.arange(n_days)
    phase = 2 * np.pi * (days - mid_winter) / 365.0
    amp  = (mean_summer - mean_winter) / 2.0
    base = (mean_summer + mean_winter) / 2.0
    return (base - amp * np.cos(phase) + rng.normal(0, noise_std, n_days)).astype(float)


def _dynamic_steady_state(temps_range: np.ndarray, amplitude: float = 5.0) -> np.ndarray:
    """Quasi-steady-state daily chill portions per temperature (slow loop)."""
    out = []
    for t in temps_range:
        ts = np.full(60, float(t))
        out.append(float(func_dynamic_chill_daily(ts, amplitude=amplitude)[-10:].mean()))
    return np.array(out)


# ---------------------------------------------------------------------------
# Season computation
# ---------------------------------------------------------------------------

def _compute_season(
    temps: np.ndarray,
    chill_fn,
    t_base: float,
    threshold_c: float,
    threshold_f: float,
) -> dict:
    n = len(temps)
    chill_daily = chill_fn(temps) if chill_fn is not None else np.zeros(n)
    force_daily = np.maximum(temps - t_base, 0.0)
    chill_cum   = np.cumsum(chill_daily)

    if chill_fn is not None:
        idx = np.where(chill_cum >= threshold_c)[0]
        fulfill = int(idx[0]) if len(idx) else n
    else:
        fulfill = 0

    force_masked = force_daily.copy()
    force_masked[:fulfill] = 0.0
    force_cum = np.cumsum(force_masked)

    idx2 = np.where(force_cum >= threshold_f)[0]
    flower = int(idx2[0]) if len(idx2) else n

    return dict(
        chill_daily=chill_daily, force_daily=force_daily,
        force_masked=force_masked, chill_cum=chill_cum, force_cum=force_cum,
        fulfill=fulfill, flower=flower, occurred=flower < n,
        threshold_c=threshold_c, threshold_f=threshold_f,
    )


MODELS = {
    'GDD': dict(
        chill_fn=None, t_base=5.0, threshold_c=0.0, threshold_f=220.0,
    ),
    'Utah + GDD': dict(
        chill_fn=func_utah_chill, t_base=5.0, threshold_c=55.0, threshold_f=220.0,
    ),
    'ChillingDays + GDD': dict(
        chill_fn=lambda ts: func_chilling_days(ts, t_threshold=7.2),
        t_base=5.0, threshold_c=55.0, threshold_f=220.0,
    ),
    'Dynamic + GDD': dict(
        chill_fn=lambda ts: func_dynamic_chill_daily(ts, amplitude=5.0),
        t_base=5.0, threshold_c=35.0, threshold_f=220.0,
    ),
}


# ---------------------------------------------------------------------------
# Bar colour helper  (mirrors model_exploration.ipynb)
# ---------------------------------------------------------------------------

def _bar_colours(chill_daily, force_daily, fulfill):
    n = len(chill_daily)
    ca   = np.abs(chill_daily[:fulfill])
    cmax = ca.max() if len(ca) > 0 and ca.max() > 0 else 1.0
    fa   = force_daily[fulfill:]
    fmax = fa.max() if len(fa) > 0 and fa.max() > 0 else 1.0
    cols = []
    for i in range(n):
        if i < fulfill:
            v = chill_daily[i]
            if v > 0.01:
                cols.append(cm.Blues(0.28 + 0.62 * v / cmax))
            elif v < -0.01:
                cols.append((0.78, 0.1, 0.1, 0.30 + 0.55 * abs(v) / cmax))
            else:
                cols.append(mcolors.to_rgba(PALETTE['none']))
        else:
            v = force_daily[i]
            if v > 0.01:
                cols.append(cm.Oranges(0.28 + 0.62 * v / fmax))
            else:
                cols.append(mcolors.to_rgba(PALETTE['none']))
    return cols


# ---------------------------------------------------------------------------
# Figure 1 — season dynamics  (two scenarios side-by-side)
# ---------------------------------------------------------------------------

def make_season_dynamics() -> plt.Figure:
    """3-row × 2-column panel: temperature bars coloured by contribution,
    cumulative chill, cumulative forcing — for a cold and a warm winter."""

    scenarios = [
        (2.0,  'Cold winter  (2 °C mean)'),
        (9.0,  'Mild winter  (9 °C mean)'),
    ]
    n_days = 270
    days   = np.arange(n_days)

    # Use Utah+GDD as the representative CF model for the bar colouring
    m = MODELS['Utah + GDD']

    fig = plt.figure(figsize=(15, 10), facecolor=PALETTE['bg'])
    fig.suptitle(
        'Chilling & forcing dynamics across contrasting winters',
        fontsize=14, fontweight='bold', y=0.98,
    )

    outer = gridspec.GridSpec(
        1, 2, wspace=0.07, left=0.07, right=0.97, top=0.93, bottom=0.07,
    )

    for col_idx, (mean_w, scenario_label) in enumerate(scenarios):
        temps = _generate_season(mean_winter=mean_w, seed=col_idx * 7 + 42)
        resp  = _compute_season(temps, m['chill_fn'], m['t_base'],
                                m['threshold_c'], m['threshold_f'])
        bar_cols = _bar_colours(resp['chill_daily'], resp['force_daily'], resp['fulfill'])
        n = len(temps)

        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[col_idx],
            hspace=0.06, height_ratios=[2.2, 1.5, 1.5],
        )
        ax_t = fig.add_subplot(inner[0])
        ax_c = fig.add_subplot(inner[1])
        ax_f = fig.add_subplot(inner[2])

        is_left  = col_idx == 0
        is_right = col_idx == 1

        # ── Panel 1: temperature bars coloured by contribution ──────────────
        ax_t.bar(days, np.abs(temps), width=1.0, color=bar_cols,
                 align='center', zorder=2)
        ax_t.plot(days, temps, color=PALETTE['temp_line'],
                  lw=1.2, zorder=3, alpha=0.85)
        ax_t.axhline(0, color='#999999', lw=0.8, zorder=1)

        cf = resp['fulfill']
        fd = resp['flower']
        if cf < n:
            ax_t.axvline(cf, color=PALETTE['chill'], lw=1.5, ls='--', alpha=0.9, zorder=4)
        if resp['occurred']:
            ax_t.axvline(fd, color=PALETTE['force'], lw=2.0, ls='-', alpha=0.95, zorder=5)
            ax_t.annotate(
                f'bloom\nday {fd}', xy=(fd, ax_t.get_ylim()[1] if ax_t.get_ylim()[1] > 0 else 15),
                xytext=(fd + 6, temps.max() * 0.85),
                fontsize=8, color=PALETTE['force'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=PALETTE['force'], lw=1.0),
            )

        legend_patches = [
            mpatches.Patch(color=cm.Blues(0.65),   label='Chill contribution'),
            mpatches.Patch(color=cm.Oranges(0.65), label='Forcing contribution'),
            mpatches.Patch(color=(0.78, 0.1, 0.1, 0.55), label='De-chill (Utah)'),
        ]
        ax_t.legend(handles=legend_patches, fontsize=8, loc='upper right',
                    framealpha=0.88, edgecolor='#cccccc')
        ax_t.set_title(scenario_label, fontsize=11, fontweight='bold', pad=6)
        _style_time_ax(ax_t, n, ylabel='Temperature (°C)' if is_left else '')
        ax_t.set_facecolor(PALETTE['bg'])

        # ── Panel 2: cumulative chill curves  (3 CF models) ──────────────
        for mname, mc in MODELS.items():
            if mc['chill_fn'] is None:
                continue
            r = _compute_season(temps, mc['chill_fn'], mc['t_base'],
                                mc['threshold_c'], mc['threshold_f'])
            norm_c = np.clip(r['chill_cum'] / mc['threshold_c'], 0, 1.05)
            color  = MODEL_COLORS[mname]
            ax_c.plot(days, norm_c, color=color, lw=1.8,
                      label=f'{mname}  (gate day {r["fulfill"] if r["fulfill"] < n else "—"})')
            if r['fulfill'] < n:
                ax_c.axvline(r['fulfill'], color=color, lw=1.0, ls=':', alpha=0.65)

        ax_c.axhline(1.0, color=PALETTE['thresh'], lw=1.0, ls='--', alpha=0.7)
        ax_c.annotate('  chilling\n  requirement', xy=(0, 1.0), xycoords=('data', 'data'),
                      fontsize=7.5, color=PALETTE['thresh'], va='center')
        ax_c.set_ylim(-0.04, 1.18)
        _style_time_ax(ax_c, n, ylabel='Chill fulfilled' if is_left else '')
        ax_c.set_yticks([0, 0.5, 1.0])
        ax_c.set_yticklabels(['0', '0.5', '1.0'], fontsize=8)
        ax_c.legend(fontsize=7.5, loc='upper left', framealpha=0.88,
                    edgecolor='#cccccc')
        ax_c.set_facecolor(PALETTE['bg'])

        # ── Panel 3: cumulative forcing curves  (all 4 models) ───────────
        for mname, mc in MODELS.items():
            r = _compute_season(temps, mc['chill_fn'], mc['t_base'],
                                mc['threshold_c'], mc['threshold_f'])
            norm_f = np.clip(r['force_cum'] / mc['threshold_f'], 0, 1.05)
            color  = MODEL_COLORS[mname]
            ls = '-' if mc['chill_fn'] is not None else '--'
            ax_f.plot(days, norm_f, color=color, lw=1.8, ls=ls,
                      label=mname + ('' if r['occurred'] else '  ✗'))
            if r['occurred']:
                ax_f.axvline(r['flower'], color=color, lw=1.0, ls=':', alpha=0.65)

        ax_f.axhline(1.0, color=PALETTE['thresh'], lw=1.0, ls='--', alpha=0.7)
        ax_f.annotate('  forcing\n  requirement', xy=(0, 1.0), xycoords=('data', 'data'),
                      fontsize=7.5, color=PALETTE['thresh'], va='center')
        ax_f.set_ylim(-0.04, 1.18)
        _style_time_ax(ax_f, n, ylabel='Forcing fulfilled' if is_left else '',
                       xlabel=True)
        ax_f.set_yticks([0, 0.5, 1.0])
        ax_f.set_yticklabels(['0', '0.5', '1.0'], fontsize=8)
        ax_f.legend(fontsize=7.5, loc='upper left', framealpha=0.88,
                    edgecolor='#cccccc')
        ax_f.set_facecolor(PALETTE['bg'])

    fig.set_facecolor(PALETTE['bg'])
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — temperature response functions
# ---------------------------------------------------------------------------

def make_chill_response() -> plt.Figure:
    """2-panel: chilling model response curves (left) and GDD forcing (right)."""
    temp_range = np.linspace(-5, 26, 400)

    utah_r  = func_utah_chill(temp_range)
    cd_r    = func_chilling_days(temp_range, t_threshold=7.2)
    dyn_r5  = _dynamic_steady_state(np.linspace(-5, 26, 80), amplitude=5.0)
    dyn_r10 = _dynamic_steady_state(np.linspace(-5, 26, 80), amplitude=10.0)
    t_dyn   = np.linspace(-5, 26, 80)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=PALETTE['bg'])
    fig.suptitle('Temperature response functions', fontsize=13, fontweight='bold', y=1.01)

    # ── Left: chilling models ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PALETTE['bg'])

    # Utah: fill positive/negative regions
    ax.fill_between(temp_range, utah_r, 0, where=(utah_r > 0),
                    color=MODEL_COLORS['Utah + GDD'], alpha=0.12)
    ax.fill_between(temp_range, utah_r, 0, where=(utah_r < 0),
                    color=PALETTE['dechill'], alpha=0.12)
    ax.axhline(0, color=PALETTE['zero'], lw=0.8)
    ax.axvline(0, color=PALETTE['zero'], lw=0.8, ls=':')

    ax.plot(temp_range, utah_r, color=MODEL_COLORS['Utah + GDD'],
            lw=2.3, label='Utah (daily, non-monotonic)')
    ax.plot(temp_range, cd_r, color=MODEL_COLORS['ChillingDays + GDD'],
            lw=2.3, label='Chilling days  (≤ 7.2 °C)')
    ax.plot(t_dyn, dyn_r5,  color=MODEL_COLORS['Dynamic + GDD'],
            lw=2.3, label='Dynamic  (amplitude = 5 °C)')
    ax.plot(t_dyn, dyn_r10, color=MODEL_COLORS['Dynamic + GDD'],
            lw=2.3, ls='--', alpha=0.55, label='Dynamic  (amplitude = 10 °C)')

    # Utah bin boundary annotations
    boundaries = [1.4, 2.4, 9.1, 12.4, 15.9, 18.0]
    bin_vals   = [0.0, 0.5, 1.0, 0.5,  0.0, -0.5, -1.0]
    t_centres  = [([(-5)] + boundaries)[i] + ((boundaries + [26])[i] - ([(-5)] + boundaries)[i]) / 2
                  for i in range(len(bin_vals))]
    for tc, val in zip(t_centres, bin_vals):
        y = func_utah_chill(np.array([tc]))[0]
        ax.text(tc, y + 0.06 * (1 if val >= 0 else -1), str(val),
                fontsize=6.5, color=MODEL_COLORS['Utah + GDD'], ha='center', alpha=0.8)
    for b in boundaries:
        ax.axvline(b, color=MODEL_COLORS['Utah + GDD'], lw=0.5, ls=':', alpha=0.4)

    ax.set_xlabel('Daily mean temperature (°C)', fontsize=10)
    ax.set_ylabel('Chill units / day', fontsize=10)
    ax.set_title('Chilling models', fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Right: GDD forcing ────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(PALETTE['bg'])

    gdd_color = MODEL_COLORS['GDD']
    for t_base, alpha, lbl in [(3, 0.45, 't_base = 3 °C'),
                                (5, 1.00, 't_base = 5 °C'),
                                (7, 0.45, 't_base = 7 °C')]:
        gdu = np.maximum(temp_range - t_base, 0.0)
        ax.fill_between(temp_range, gdu, 0, color=PALETTE['force'], alpha=0.06 * (2 - alpha + 1))
        ax.plot(temp_range, gdu, color=PALETTE['force'],
                lw=2.2, alpha=alpha, label=lbl)
    ax.axvline(5, color=PALETTE['force'], lw=0.7, ls=':', alpha=0.5)
    ax.set_xlabel('Daily mean temperature (°C)', fontsize=10)
    ax.set_ylabel('GDU / day  (forcing units)', fontsize=10)
    ax.set_title('Forcing model  —  GDU = max(T − t_base, 0)', fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — winter warmth sensitivity
# ---------------------------------------------------------------------------

def make_winter_sensitivity() -> plt.Figure:
    """Bloom DOY and chilling-gate day vs mean winter temperature, all models."""
    winter_temps = np.arange(0.0, 16.5, 0.5)
    n_seeds = 12
    n_days  = 270

    results: dict[str, dict[str, list]] = {
        name: {'bloom': [], 'bloom_lo': [], 'bloom_hi': [],
               'chill': [], 'chill_lo': [], 'chill_hi': []}
        for name in MODELS
    }

    for mw in winter_temps:
        for name, m in MODELS.items():
            blooms, chills = [], []
            for seed in range(n_seeds):
                ts = _generate_season(mean_winter=mw, seed=seed * 3 + 1)
                r  = _compute_season(ts, m['chill_fn'], m['t_base'],
                                     m['threshold_c'], m['threshold_f'])
                blooms.append(r['flower'] if r['occurred'] else np.nan)
                chills.append(r['fulfill'] if r['fulfill'] < n_days else np.nan)
            results[name]['bloom'].append(np.nanmedian(blooms))
            results[name]['bloom_lo'].append(np.nanpercentile(blooms, 20) if not all(np.isnan(blooms)) else np.nan)
            results[name]['bloom_hi'].append(np.nanpercentile(blooms, 80) if not all(np.isnan(blooms)) else np.nan)
            results[name]['chill'].append(np.nanmedian(chills))
            results[name]['chill_lo'].append(np.nanpercentile(chills, 20) if not all(np.isnan(chills)) else np.nan)
            results[name]['chill_hi'].append(np.nanpercentile(chills, 80) if not all(np.isnan(chills)) else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=PALETTE['bg'])
    fig.suptitle(
        'Model sensitivity to winter warmth — synthetic climate scenarios',
        fontsize=13, fontweight='bold', y=1.02,
    )

    # ── Left: bloom DOY vs winter temperature ──────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PALETTE['bg'])
    for name, color in MODEL_COLORS.items():
        med = np.array(results[name]['bloom'])
        lo  = np.array(results[name]['bloom_lo'])
        hi  = np.array(results[name]['bloom_hi'])
        mask = ~np.isnan(med)
        if mask.any():
            ax.plot(winter_temps[mask], med[mask], color=color, lw=2.2, label=name)
            ax.fill_between(winter_temps[mask], lo[mask], hi[mask],
                            color=color, alpha=0.15)
        # Mark the first NaN (chilling failure)
        if not mask.all():
            fail_idx = np.where(~mask)[0]
            if len(fail_idx):
                ax.axvline(winter_temps[fail_idx[0]], color=color,
                           lw=1.0, ls=':', alpha=0.6)

    ax.set_xlabel('Mean winter temperature (°C)', fontsize=10)
    ax.set_ylabel('Predicted bloom  (day of season)', fontsize=10)
    ax.set_title('Bloom timing', fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Right: chilling gate day vs winter temperature ─────────────────────
    ax = axes[1]
    ax.set_facecolor(PALETTE['bg'])
    for name, color in MODEL_COLORS.items():
        if MODELS[name]['chill_fn'] is None:
            continue
        med = np.array(results[name]['chill'])
        lo  = np.array(results[name]['chill_lo'])
        hi  = np.array(results[name]['chill_hi'])
        mask = ~np.isnan(med)
        if mask.any():
            ax.plot(winter_temps[mask], med[mask], color=color, lw=2.2, label=name)
            ax.fill_between(winter_temps[mask], lo[mask], hi[mask],
                            color=color, alpha=0.15)
        if not mask.all():
            fail_idx = np.where(~mask)[0]
            if len(fail_idx):
                ax.axvline(winter_temps[fail_idx[0]], color=color,
                           lw=1.0, ls=':', alpha=0.6)
                ax.annotate(
                    f'{name.split()[0]}\nfails here',
                    xy=(winter_temps[fail_idx[0]], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 200),
                    xytext=(winter_temps[fail_idx[0]] + 0.4, 210 - list(MODEL_COLORS.keys()).index(name) * 18),
                    fontsize=7, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                )

    ax.set_xlabel('Mean winter temperature (°C)', fontsize=10)
    ax.set_ylabel('Day chilling requirement met', fontsize=10)
    ax.set_title('Chilling gate opening day', fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Generate README figures and save to outputs/readme/.',
    )
    parser.add_argument('--formats', nargs='+', default=['png'], metavar='FMT')
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s  %(message)s',
    )

    out_dir = get_repo_root() / 'outputs' / 'readme'
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(args.formats)

    figures = [
        ('season_dynamics',   make_season_dynamics),
        ('chill_response',    make_chill_response),
        ('winter_sensitivity', make_winter_sensitivity),
    ]

    for stem, fn in figures:
        log.info("Generating %s ...", stem)
        try:
            fig = fn()
            save_figure(fig, out_dir, stem, formats=formats, dpi=args.dpi)
            plt.close(fig)
            log.info("  Saved  %s/%s", out_dir.name, stem)
        except Exception:
            log.error("%s failed:\n%s", stem, traceback.format_exc())

    log.info("Done. Figures saved under %s", out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
