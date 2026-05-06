"""
run.py

Single/multi-run entrypoint for RHEA-NL (YAML-driven).

Key design:
- A YAML config provides:
    (i) paths (dataset + results directory)
    (ii) one or more named scenarios (parameter sets)
    (iii) OPTIONAL replication controls per scenario:
        - n_runs: int (default 1)
        - base_seed: int (default: random_seed if provided else 0)
        - seed_step: int (default 1)

Usage
-----
python run.py --config single_run_scenarios.yaml --scenario S1d

If you omit --scenario, we default to the first scenario key in the YAML (excluding _paths).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Tuple, List

import yaml
import pandas as pd

from model import RHEA_Model


# ---------------------------------------------------------------------
# YAML loading / normalization
# ---------------------------------------------------------------------


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict, got {type(data)}")
    return data


def _resolve_paths(cfg_dir: str, paths_block: Dict[str, Any]) -> Tuple[str, str]:
    """Resolve dataset and results paths.

    - If dataset_csv / results_dir are absolute, keep them.
    - If they are relative, resolve relative to the YAML directory.
    """
    dataset_csv = str(paths_block.get("dataset_csv"))
    results_dir = str(paths_block.get("results_dir"))

    if not os.path.isabs(dataset_csv):
        dataset_csv = os.path.normpath(os.path.join(cfg_dir, dataset_csv))
    if not os.path.isabs(results_dir):
        results_dir = os.path.normpath(os.path.join(cfg_dir, results_dir))

    return dataset_csv, results_dir


def _pick_scenario(cfg: Dict[str, Any], scenario_name: str | None) -> Tuple[str, Dict[str, Any]]:
    """Return (scenario_key, scenario_dict)."""
    scenario_keys = [k for k in cfg.keys() if k != "_paths"]
    if not scenario_keys:
        raise KeyError("No scenarios found in YAML (expected at least one top-level key besides '_paths').")

    if scenario_name is None:
        key = scenario_keys[0]
        return key, dict(cfg[key])

    if scenario_name not in cfg:
        raise KeyError(f"Scenario '{scenario_name}' not found. Available: {scenario_keys}")

    return scenario_name, dict(cfg[scenario_name])


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def _setup_logging(results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-runs (e.g., notebook execution)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(
        os.path.join(results_dir, "run.log"),
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def log_snapshot(model: Any, step: int) -> None:
    """Log a compact per-step snapshot without assuming exact internal names."""
    parcels = getattr(model, "parcels", [])
    households = getattr(model, "households", getattr(model, "agents", []))
    buyers = getattr(model, "active_buyers", getattr(model, "buyers", []))
    sellers = getattr(model, "active_sellers", getattr(model, "sellers", []))
    tx = getattr(model, "transactions", {})

    n_parcels = _safe_len(parcels)
    n_hh = _safe_len(households)
    n_buyers = _safe_len(buyers)
    n_sellers = _safe_len(sellers)
    n_tx = _safe_len(tx) if isinstance(tx, dict) else _safe_len(getattr(tx, "items", []))

    realtor = getattr(model, "realtor", None)
    r2 = nobs = k_used = None
    if getattr(realtor, "result", None) is not None:
        try:
            r2 = float(realtor.result.rsquared)
            nobs = int(realtor.result.nobs)
            k_used = getattr(realtor, "last_used_k", None)
        except Exception:
            pass

    logging.info(
        "Step %4d | Parcels=%d | Households=%d | Buyers=%d | Sellers=%d | Transactions=%d%s",
        step,
        n_parcels,
        n_hh,
        n_buyers,
        n_sellers,
        n_tx,
        ("" if r2 is None else f" | Realtor: R2={r2:.3f} nobs={nobs} k={k_used}"),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RHEA-NL simulation(s) from a YAML scenario config.")
    ap.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "single_run_scenarios.yaml"),
        help="Path to single-run scenarios YAML.",
    )
    ap.add_argument(
        "--scenario",
        default=None,
        help="Scenario key in the YAML (defaults to the first scenario).",
    )
    ap.add_argument(
        "--snapshot_every",
        type=int,
        default=5,
        help="Log a snapshot every N steps (0 disables periodic snapshots).",
    )
    ap.add_argument(
        "--combine",
        action="store_true",
        help="If set, also save concatenated model/agent outputs across runs (batch-style).",
    )
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg_dir = os.path.dirname(cfg_path)

    cfg = _load_yaml(cfg_path)
    if "_paths" not in cfg:
        raise KeyError("YAML must contain a top-level '_paths' block.")

    dataset_csv, base_results_dir = _resolve_paths(cfg_dir, cfg["_paths"])
    scenario_key, scenario = _pick_scenario(cfg, args.scenario)

    # Replication controls (passed via YAML scenario)
    n_runs = int(scenario.get("n_runs", 1))
    seed_step = int(scenario.get("seed_step", 1))

    # If scenario sets random_seed, use that as default base_seed; else 0
    base_seed = int(scenario.get("base_seed", scenario.get("random_seed", 0)))

    # Create a scenario-specific run folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_results_dir, f"single_{scenario_key}_{stamp}")
    _setup_logging(results_dir)

    logging.info("Config: %s", cfg_path)
    logging.info("Scenario: %s", scenario_key)
    logging.info("Dataset: %s", dataset_csv)
    logging.info("Results: %s", results_dir)
    logging.info("Replications: n_runs=%d | base_seed=%d | seed_step=%d", n_runs, base_seed, seed_step)

    # Model kwargs: YAML scenario overrides + required fields
    model_kwargs_base = dict(scenario)
    model_kwargs_base["parcel_file"] = dataset_csv

    # Remove replication-only keys so model init doesn't accidentally accept them
    for k in ("n_runs", "base_seed", "seed_step"):
        model_kwargs_base.pop(k, None)

    # Sanity: ensure kY/years exist so we can compute total steps
    model_kwargs_base.setdefault("kY", 2)
    model_kwargs_base.setdefault("years", 30)

    # Save the exact run config next to outputs (reproducibility)
    resolved = {
        "_paths": {"dataset_csv": dataset_csv, "results_dir": base_results_dir},
        scenario_key: dict(scenario),
    }
    with open(os.path.join(results_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    total_steps = int(model_kwargs_base["kY"]) * int(model_kwargs_base["years"])

    all_mv: List[pd.DataFrame] = []
    all_av: List[pd.DataFrame] = []

    logging.info("Starting %d run(s)…", n_runs)
    t0_all = time.time()

    for run_id in range(n_runs):
        seed = base_seed + run_id

        run_kwargs = dict(model_kwargs_base)
        run_kwargs["random_seed"] = seed

        logging.info("Run %d/%d | seed=%d | starting…", run_id + 1, n_runs, seed)
        t0 = time.time()

        model = RHEA_Model(**run_kwargs)
        log_snapshot(model, step=0)

        for step in range(total_steps):
            model.step(run_number=run_id)
            if args.snapshot_every and (step + 1) % args.snapshot_every == 0:
                log_snapshot(model, step=step + 1)

        elapsed = time.time() - t0
        logging.info("Run %d finished in %.2f seconds (%.2f minutes).", run_id, elapsed, elapsed / 60.0)

        # Save outputs (seed-suffixed like old batch outputs)
        mv = model.datacollector.get_model_vars_dataframe()
        av = model.datacollector.get_agent_vars_dataframe()

        mv.insert(0, "Run", run_id)
        av.insert(0, "Run", run_id)

        mv_path = os.path.join(results_dir, f"model_vars_seed{seed}.pkl")
        av_path = os.path.join(results_dir, f"agent_vars_seed{seed}.pkl")
        mv.to_pickle(mv_path)
        av.to_pickle(av_path)
        model.export_income_statistics(os.path.join(results_dir, f"income_statistics_seed{seed}.csv"))

    elapsed_all = time.time() - t0_all
    logging.info("All runs finished in %.2f seconds (%.2f minutes).", elapsed_all, elapsed_all / 60.0)
    logging.info("Results saved to: %s", results_dir)


if __name__ == "__main__":
    main()