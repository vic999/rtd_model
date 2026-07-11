#!/usr/bin/env python3
"""
Command-line interface for the continuous virus-filtration RTD model.

Experiments are configured in ``experiments.yaml`` (add your own there — the
CLI discovers them automatically). Four sub-commands:

    list      list all configured experiments (C* and V*)
    figure    build the Figure 3 / Figure 4 multi-panel grids
    plot      build a high-resolution plot for one or more experiments
    csv       export the full simulated data for experiments as CSV

Run ``python3 rtd_cli.py -h`` or ``python3 rtd_cli.py <command> -h`` for help.

Examples
--------
    python3 rtd_cli.py list
    python3 rtd_cli.py list --figure 3
    python3 rtd_cli.py figure --which both --style paper
    python3 rtd_cli.py plot --experiment C1 V2-3 --dpi 300
    python3 rtd_cli.py plot --experiment all --out plots
    python3 rtd_cli.py csv --experiment C3-3 --out data
"""

from __future__ import annotations

import argparse
import os
import sys

from rtd.experiments import load_config, simulate, find_experiment
from rtd.plots import plot_grid, plot_single, export_csv


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _select(experiments, names):
    """Resolve a list of names (or ['all']) to Experiment objects."""
    if not names or [n.lower() for n in names] == ["all"]:
        return list(experiments)
    return [find_experiment(experiments, n) for n in names]


def _ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    return d or "."


def _defget(defaults, key, fallback):
    v = defaults.get(key, fallback)
    return fallback if v is None else v


# --------------------------------------------------------------------------
# sub-commands
# --------------------------------------------------------------------------
def cmd_list(args, experiments, defaults):
    rows = [e for e in experiments if args.figure in (None, e.figure)]
    if not rows:
        print("no experiments match."); return
    w = max(len(e.name) for e in rows)
    print(f"{'name'.ljust(w)}  fig  kind   connection  flow(mL/min)  surf(cm2)  description")
    print("-" * 78)
    for e in rows:
        surf = "-" if e.surface is None else str(e.surface)
        fig = "-" if e.figure is None else str(e.figure)
        print(f"{e.name.ljust(w)}  {fig:>3}  {e.display_kind:<5}  {e.connection:<10}  "
              f"{e.flow:>11}  {surf:>8}  {e.description}")
    print(f"\n{len(rows)} experiment(s). Config: {args.config or 'experiments.yaml'}")


def cmd_figure(args, experiments, defaults):
    style = args.style or _defget(defaults, "style", "paper")
    focus = _defget(defaults, "focus", False) if args.focus is None else args.focus
    dpi = args.dpi or _defget(defaults, "dpi", 130)
    n_time = _defget(defaults, "n_time", 1400)
    outdir = _ensure_dir(args.out)

    which = [3, 4] if args.which == "both" else [int(args.which)]
    titles = {
        3: ("Figure 3 (reproduced): pulse-injection calibration curves\n"
            "UV via Beer's law, conductivity via Kohlrausch's law"),
        4: ("Figure 4 (reproduced): stepwise validation curves\n"
            "start-up plateau + wash-out"),
    }
    for fno in which:
        exps = [e for e in experiments if e.figure == fno]
        if not exps:
            print(f"figure {fno}: no experiments configured, skipping."); continue
        pairs = [(e, simulate(e, n_time=n_time)) for e in exps]
        out = os.path.join(outdir, f"figure{fno}.png")
        plot_grid(pairs, titles[fno], out, style=style, focus=focus, dpi=dpi)
        print("wrote", out)


def cmd_plot(args, experiments, defaults):
    style = args.style or _defget(defaults, "style", "paper")
    focus = _defget(defaults, "focus", False) if args.focus is None else args.focus
    dpi = args.dpi or _defget(defaults, "single_dpi", 300)
    size = args.size or _defget(defaults, "single_size", [9.0, 6.0])
    n_time = _defget(defaults, "n_time", 1400)
    outdir = _ensure_dir(args.out)

    for e in _select(experiments, args.experiment):
        sim = simulate(e, n_time=n_time)
        out = os.path.join(outdir, f"{e.name}.png")
        plot_single(e, sim, out, style=style, focus=focus, dpi=dpi,
                    figsize=tuple(size))
        print("wrote", out)


def cmd_csv(args, experiments, defaults):
    n_time = args.n_time or _defget(defaults, "n_time", 1400)
    outdir = _ensure_dir(args.out)
    for e in _select(experiments, args.experiment):
        sim = simulate(e, n_time=n_time)
        out = os.path.join(outdir, f"{e.name}.csv")
        export_csv(e, sim, out)
        print("wrote", out)


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="rtd_cli.py",
        description="RTD model for continuous virus filtration — reproduce "
                    "Chen et al. (2024). Experiments are configured in "
                    "experiments.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 rtd_cli.py list\n"
               "  python3 rtd_cli.py figure --which 3 --style paper\n"
               "  python3 rtd_cli.py plot --experiment C1 V2-3 --dpi 300\n"
               "  python3 rtd_cli.py plot --experiment all --out plots\n"
               "  python3 rtd_cli.py csv  --experiment all --out data\n")
    p.add_argument("--config", metavar="FILE",
                   help="experiments YAML (default: experiments.yaml)")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    pl = sub.add_parser("list", help="list all configured experiments")
    pl.add_argument("--figure", type=int, choices=[3, 4],
                    help="only show experiments for this figure")
    pl.set_defaults(func=cmd_list)

    pf = sub.add_parser("figure", help="build Figure 3 / 4 multi-panel grids")
    pf.add_argument("--which", choices=["3", "4", "both"], default="both",
                    help="which figure(s) to build (default: both)")
    pf.add_argument("--style", choices=["paper", "overlay"],
                    help="panel layout (default: from YAML / paper)")
    pf.add_argument("--focus", action=argparse.BooleanOptionalAction,
                    help="tight x-axis auto-crop (--focus / --no-focus)")
    pf.add_argument("--dpi", type=int, help="image DPI (default: from YAML / 130)")
    pf.add_argument("--out", metavar="DIR", default=".", help="output directory")
    pf.set_defaults(func=cmd_figure)

    pp = sub.add_parser("plot", help="high-resolution plot per experiment")
    pp.add_argument("--experiment", nargs="+", required=True, metavar="NAME",
                    help="experiment name(s), or 'all'")
    pp.add_argument("--style", choices=["paper", "overlay"],
                    help="panel layout (default: from YAML / paper)")
    pp.add_argument("--focus", action=argparse.BooleanOptionalAction,
                    help="tight x-axis auto-crop (--focus / --no-focus)")
    pp.add_argument("--dpi", type=int,
                    help="image DPI (default: from YAML / 300)")
    pp.add_argument("--size", nargs=2, type=float, metavar=("W", "H"),
                    help="figure size in inches (default: from YAML / 9 6)")
    pp.add_argument("--out", metavar="DIR", default=".", help="output directory")
    pp.set_defaults(func=cmd_plot)

    pc = sub.add_parser("csv", help="export simulated data as CSV")
    pc.add_argument("--experiment", nargs="+", required=True, metavar="NAME",
                    help="experiment name(s), or 'all'")
    pc.add_argument("--n-time", type=int, dest="n_time",
                    help="number of time samples (default: from YAML / 1400)")
    pc.add_argument("--out", metavar="DIR", default=".", help="output directory")
    pc.set_defaults(func=cmd_csv)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        experiments, defaults = load_config(args.config)
        args.func(args, experiments, defaults)
    except FileNotFoundError as e:
        parser.exit(2, f"error: config file not found: {e.filename}\n")
    except (KeyError, ValueError) as e:
        parser.exit(2, f"error: {str(e).strip(chr(34))}\n")


if __name__ == "__main__":
    main()
