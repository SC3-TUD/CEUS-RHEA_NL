# -*- coding: utf-8 -*-
"""
realtor.py

@author: asli-mutlu

Realtor agent for RHEA-NL.

ROLE
----
1) Fit a rolling hedonic OLS on recent simulated transactions using EXACTLY:
   ['AGE','HOUSESIZE','LOTSIZE','ROOMS','QUALITY','LN_DIST_CBD','LN_DIST_MEUSE','d_floodprone'].
2) Use the model to predict market prices for parcels each step (vectorized when possible).
3) Fall back to each parcel's `last_transaction_price` when no model is available or prediction fails.

KEY CHOICES (behavior-preserving)
--------------------------------
- Rolling window k = min_steps..max_steps; accept smallest k where at least half of non-constant
  coefficients are significant (p < pval_threshold).
- OLS is fit on log(P_trans); predictions are returned in levels via exp().
- Predicted prices are rounded to the nearest 100 euros, with a floor at 100 euros.
- IMPORTANT: When building the estimation sample for a window, if the same parcel appears in
  multiple step-dictionaries (sold multiple times across the window), we keep ONLY the most
  recent sale for that parcel (i.e., a unique-parcel cross-section per window).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from mesa import Agent
except Exception:
    class Agent:  # type: ignore
        def __init__(self, unique_id, model):
            self.unique_id = unique_id
            self.model = model


class Realtor(Agent):
    """
    Mesa Agent implementing a rolling OLS for price estimation with fixed regressors.

    Expected model contract (CURRENT):
        - model.transactions: dict[prop_id(int) -> dict], containing at least {"P_trans": float}
        - model.parcels: iterable of Parcel objects (each with .unique_id and .get_prop_chars())
    """

    def __init__(
        self,
        unique_id,
        model,
        min_steps: int = 4,
        max_steps: int = 10,
        pval_threshold: float = 0.10,
        price_rounding: int = -2,  # -2 means nearest 100
        min_price_floor: float = 1e2,
    ):
        super().__init__(unique_id, model)

        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.pval_threshold = float(pval_threshold)

        self.price_rounding = int(price_rounding)
        self.min_price_floor = float(min_price_floor)

        self.ind_vars: Tuple[str, ...] = (
            "AGE",
            "HOUSESIZE",
            "LOTSIZE",
            "ROOMS",
            "QUALITY",
            "LN_DIST_CBD",
            "LN_DIST_MEUSE",
            "d_floodprone",
        )

        self.result: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
        self.exog_names: Optional[Iterable[str]] = None
        self.last_used_k: int = 0
        self.last_fit_step: Optional[int] = None

        # Each entry is a per-step dict: {prop_id(int) -> record_dict}
        self.market_history: List[Dict[int, Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Helpers: parcel keys and resolver
    # ------------------------------------------------------------------

    def _parcel_key(self, p: Any) -> int:
        uid = getattr(p, "unique_id", None)
        return int(uid) if uid is not None else id(p)

    def _build_id_to_parcel(self) -> Dict[int, Any]:
        parcels = getattr(self.model, "parcels", None)
        if parcels is None:
            return {}
        # model.parcels can be list or dict; support both
        if isinstance(parcels, dict):
            it = parcels.values()
        else:
            it = parcels
        return {self._parcel_key(p): p for p in it}

    # ------------------------------------------------------------------
    # Helper: estimation sample construction
    # ------------------------------------------------------------------

    def _merge_keep_latest(self, win: Sequence[Dict[int, Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """
        Merge step-level transaction dicts into one dict where each prop_id appears once.
        If a prop_id appears multiple times across the window, keep the MOST RECENT occurrence.
        """
        merged: Dict[int, Dict[str, Any]] = {}
        for step_dict in win:
            if isinstance(step_dict, dict) and step_dict:
                merged.update(step_dict)  # newer overwrites older
        return merged

    def _tx_dict_to_Xy(
        self,
        tx_dict: Dict[int, Dict[str, Any]],
        id_to_parcel: Dict[int, Any],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert {prop_id -> {"P_trans": price, ...}} to (X, y) for OLS.

        y = log(P_trans), P_trans > 0
        X = eight required features from parcel.get_prop_chars()

        Note: if a prop_id cannot be resolved to a Parcel object, it is skipped.
        """
        if not tx_dict:
            raise ValueError("Empty transactions dict.")

        prices: List[float] = []
        parcels: List[Any] = []

        for prop_id, rec in tx_dict.items():
            if not isinstance(rec, dict) or "P_trans" not in rec:
                raise ValueError("Each transaction record must be a dict containing 'P_trans'.")

            parcel = id_to_parcel.get(int(prop_id))
            if parcel is None:
                # If this happens a lot, your parcel ids are not aligned with model.parcels.unique_id
                continue

            p = float(rec["P_trans"])
            if not np.isfinite(p) or p <= 0:
                raise ValueError(f"Invalid P_trans for log(): {p}. Must be finite and > 0.")

            prices.append(p)
            parcels.append(parcel)

        if len(parcels) == 0:
            raise ValueError("No resolvable parcels in transactions (id_to_parcel lookup failed).")

        y = pd.Series(np.log(np.array(prices, dtype=float)), name="log_price")

        X_rows: List[Dict[str, float]] = []
        for parcel in parcels:
            feat = parcel.get_prop_chars()
            row: Dict[str, float] = {}
            for c in self.ind_vars:
                if c not in feat:
                    raise ValueError(f"Missing feature '{c}' for parcel {getattr(parcel, 'unique_id', None)}")
                v = float(feat[c])
                if not np.isfinite(v):
                    raise ValueError(f"Non-finite feature '{c}' for parcel {getattr(parcel, 'unique_id', None)}")
                row[c] = v
            X_rows.append(row)

        X = pd.DataFrame(X_rows, columns=list(self.ind_vars)).astype(float)
        X = sm.add_constant(X, has_constant="add")
        return X, y

    def _significance_ok(self, res: sm.regression.linear_model.RegressionResultsWrapper) -> bool:
        """At least half of non-constant coefficients significant at p < pval_threshold."""
        try:
            p = res.pvalues.reindex([c for c in res.params.index if c != "const"]).dropna()
            if len(p) == 0:
                return False
            needed = max(1, int(np.ceil(len(p) / 2)))
            return int((p < self.pval_threshold).sum()) >= needed
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_on_history(self, history: Sequence[Dict[int, Dict[str, Any]]], step_now: int):
        """Fit/update OLS using the last k dicts in history (k in [min_steps, max_steps])."""
        if not history:
            logging.info("Realtor.fit: empty history at step %s.", step_now)
            return self.result

        id_to_parcel = self._build_id_to_parcel()
        if not id_to_parcel:
            logging.warning("Realtor.fit: cannot build id_to_parcel mapping; skipping fit.")
            return self.result

        accepted = None
        used_k = None

        k_max_eff = min(self.max_steps, len(history))
        if k_max_eff <= 0:
            return self.result

        k_min_eff = min(self.min_steps, k_max_eff)

        for k in range(k_min_eff, k_max_eff + 1):
            win = history[-k:]
            merged = self._merge_keep_latest(win)
            if not merged:
                continue

            try:
                X, y = self._tx_dict_to_Xy(merged, id_to_parcel=id_to_parcel)
                if len(y) <= X.shape[1]:
                    continue
                res = sm.OLS(y, X).fit()
            except Exception as e:
                logging.warning("Realtor.fit: OLS failed at k=%s: %s", k, e)
                continue

            if self._significance_ok(res):
                accepted = res
                used_k = k
                break

        if accepted is not None:
            self.result = accepted
            self.exog_names = list(accepted.model.exog_names)
            self.last_used_k = int(used_k or 0)
            self.last_fit_step = int(step_now)
            logging.info(
                "Realtor.fit: accepted model at step %s with k=%s; nobs=%s; R2=%.3f",
                step_now, used_k, int(accepted.nobs), float(accepted.rsquared),
            )
        else:
            logging.info("Realtor.fit: no acceptable model at step %s; keeping previous model.", step_now)

        return self.result

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _fallback_price(self, parcel: Any) -> float:
        ltp = getattr(parcel, "last_transaction_price", None)
        try:
            v = float(ltp)
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    def _build_feature_matrix(self, parcels: Sequence[Any]) -> Tuple[pd.DataFrame, List[Any], List[Any]]:
        if self.exog_names is None:
            raise ValueError("Cannot build feature matrix without exog_names. Fit a model first.")

        rows: List[Dict[str, float]] = []
        keep: List[Any] = []
        skipped: List[Any] = []

        for p in parcels:
            try:
                feat = p.get_prop_chars()
                row = {c: float(feat[c]) for c in self.ind_vars}
                row["const"] = 1.0
                rows.append(row)
                keep.append(p)
            except Exception:
                skipped.append(p)

        X = pd.DataFrame(rows, columns=list(self.exog_names)) if rows else pd.DataFrame(columns=list(self.exog_names))
        return X, keep, skipped

    def _postprocess_prices(self, prices_level: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices_level, dtype=float)
        prices = np.maximum(prices, self.min_price_floor)
        prices = np.round(prices, self.price_rounding)
        return prices

    def _predict_levels_from_X(self, X: pd.DataFrame) -> np.ndarray:
        if self.result is None:
            raise ValueError("No fitted model available for prediction.")
        logp = self.result.predict(X)
        prices = np.exp(np.asarray(logp, dtype=float))
        return self._postprocess_prices(prices)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict_prices_batch(self, parcels: Sequence[Any]) -> Dict[int, float]:
        """
        Vectorized price prediction for many parcels.

        Returns:
            dict {parcel_key(int) -> price(float)} for ALL input parcels.
        """
        parcels = list(parcels)

        if self.result is None or self.exog_names is None:
            return {self._parcel_key(p): self._fallback_price(p) for p in parcels}

        X, keep, skipped = self._build_feature_matrix(parcels)

        out: Dict[int, float] = {}

        if len(keep) > 0:
            try:
                prices = self._predict_levels_from_X(X)
                for i, p in enumerate(keep):
                    out[self._parcel_key(p)] = float(prices[i])
            except Exception as e:
                logging.warning(
                    "Realtor.predict(batch): vectorized predict failed (%s); falling back per-parcel.",
                    e,
                )
                out.clear()

        if len(out) == 0:
            for p in parcels:
                out[self._parcel_key(p)] = self.predict_price(p)
            return out

        for p in skipped:
            out[self._parcel_key(p)] = self._fallback_price(p)

        if len(out) != len(parcels):
            for p in parcels:
                k = self._parcel_key(p)
                if k not in out:
                    out[k] = self._fallback_price(p)

        return out

    def predict_price(self, parcel: Any) -> float:
        if self.result is None or self.exog_names is None:
            return self._fallback_price(parcel)

        try:
            feat = parcel.get_prop_chars()
            row = {c: float(feat[c]) for c in self.ind_vars}
            row["const"] = 1.0
            X = pd.DataFrame([row], columns=list(self.exog_names))

            price = float(np.exp(self.result.predict(X)[0]))
            if not np.isfinite(price) or price <= 0:
                return self._fallback_price(parcel)
            return float(self._postprocess_prices(np.array([price]))[0])
        except Exception:
            return self._fallback_price(parcel)

    def get_regression_coefs(self) -> dict:
        if self.result is None:
            return {}
        try:
            res = self.result
            return {
                "exog": list(res.model.exog_names),
                "coef": {k: float(v) for k, v in res.params.items()},
                "se": {k: float(v) for k, v in res.bse.items()},
                "p": {k: float(v) for k, v in res.pvalues.items()},
                "nobs": int(res.nobs),
                "r2": float(res.rsquared),
                "k_used": int(self.last_used_k),
                "fit_step": int(self.last_fit_step if self.last_fit_step is not None else -1),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Mesa stages
    # ------------------------------------------------------------------

    def stage0(self):
        """Stage 0 — fit (if possible) and set market_price for all parcels."""
        step_now = getattr(self.model.schedule, "steps", None)

        if step_now is not None and len(self.market_history) > 0:
            self.fit_on_history(self.market_history, step_now=int(step_now))

        all_parcels = getattr(self.model, "parcels", None)
        if all_parcels is None:
            logging.error("Realtor.stage0: model has no 'parcels' attribute.")
            return

        # Support both list and dict storage
        parcels_iter = all_parcels.values() if isinstance(all_parcels, dict) else all_parcels

        price_map = self.predict_prices_batch(list(parcels_iter))

        parcels_iter = all_parcels.values() if isinstance(all_parcels, dict) else all_parcels
        for parcel in parcels_iter:
            k = self._parcel_key(parcel)
            mv = price_map.get(k, self._fallback_price(parcel))
            parcel.market_price = float(mv)

        logging.info(
            "Realtor.stage0: set market prices for %d parcels (regression=%s, k_used=%s)",
            len(price_map),
            "on" if self.result is not None else "off",
            self.last_used_k,
        )

    def stage1(self):
        pass

    def stage2(self):
        pass

    def stage3(self):
        pass

    def stage4(self):
        """Finalize step — append this step's trades to history (copy to avoid mutation)."""
        tx_this_step = getattr(self.model, "transactions", None)
        if isinstance(tx_this_step, dict) and len(tx_this_step) > 0:
            # Copy is crucial: model may reset/overwrite transactions dict next step.
            self.market_history.append(dict(tx_this_step))
            logging.debug(
                "Realtor.stage4: appended %d trades to history (len=%d).",
                len(tx_this_step),
                len(self.market_history),
            )
        else:
            logging.debug("Realtor.stage4: no trades to append this step.")