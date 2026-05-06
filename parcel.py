# -*- coding: utf-8 -*-
"""
@author: asli-mutlu

Parcel (dwelling) object used by RHEA-NL.

Key concepts:
- `ask_price`: what the seller lists the house for.
- `market_price`: what the market expects the house is worth (e.g., the realtor's predicted price).
- `last_transaction_price`: realized price from the last sale (initially set from input price).
- `current_price`: safe fallback used by other agents when no model-based price is available.

Flood coding:
- We standardize the flood-prone dummy to `d_floodprone` (0/1).
- The raw input key `FP_PROTECTED` is required.
"""

from __future__ import annotations

from dataclasses import dataclass 
from typing import Dict, Any, Optional

def _to_float(x: Any, field_name: str) -> float:
    """
    Convert input to float with a clear error if conversion fails.
    We accept numeric strings as well (e.g., "123.4").
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f"Parcel field '{field_name}' must be numeric, got {x!r} ({type(x)})")


def _to_int(x: Any, field_name: str) -> int:
    """
    Convert input to int with a clear error if conversion fails.
    Useful for discrete attributes like rooms.
    """
    try:
        return int(x)
    except (TypeError, ValueError):
        raise ValueError(f"Parcel field '{field_name}' must be int-like, got {x!r} ({type(x)})")


@dataclass
class Parcel:
    """
    A housing unit (parcel) in the model.

    Parameters
    ----------
    unique_id:
        A unique identifier for the parcel (int recommended).
    parcel_chars:
        Dictionary of parcel characteristics coming from the synthetic stock / data pipeline.
        Expected keys include (minimum):
            - AGE
            - HOUSESIZE
            - LOTSIZE
            - ROOMS
            - QUALITY
            - LN_DIST_CBD
            - LN_DIST_MEUSE
            - FP_PROTECTED (or an alias)
        Plus one of:
            - PREDICTED_PRICE   (preferred when available)
            - INIT_PRICE_2020   (fallback)

    Notes
    -----
    - We keep `prop_chars_raw` for traceability (useful when debugging data).
    - We also store cleaned / typed attributes for safe downstream use.
    """

    unique_id: int
    prop_chars_raw: Dict[str, Any]

    # --- Core economic state (mutable during simulation) ---
    ask_price: Optional[float] = None
    market_price: Optional[float] = None
    last_transaction_price: float = 0.0
    initial_price: float = 0.0

    # --- Market process bookkeeping (set/updated by households) ---
    N_sales: int = 0
    listing_time: int = 0
    expected_flood_loss: float = 0.0
    amenity_value: float = 0.0

    # --- Ownership (set during initialization / transactions) ---
    owner: Any = None  # typically a Household; left as Any to avoid import cycles

    # --- Cleaned parcel attributes (typed) ---
    AGE: float = 0.0
    HOUSESIZE: float = 0.0
    LOTSIZE: float = 0.0
    ROOMS: int = 0
    QUALITY: int = 0
    LN_DIST_CBD: float = 0.0
    LN_DIST_MEUSE: float = 0.0
    DIST_MEUSE: Optional[float] = None
    d_floodprone: int = 0

    # The regression feature order used by the Realtor's hedonic regression.
    # Keep this stable to avoid silent coefficient-feature mismatch.
    _FEATURE_ORDER = (
        "AGE",
        "HOUSESIZE",
        "LOTSIZE",
        "ROOMS",
        "QUALITY",
        "LN_DIST_CBD",
        "LN_DIST_MEUSE",
        "d_floodprone",
    )

    def __init__(self, unique_id: int, parcel_chars: Dict[str, Any]):
        self.unique_id = unique_id
        self.prop_chars_raw = dict(parcel_chars)  # defensive copy for reproducibility

        # --- Prices: prefer PREDICTED_PRICE; fallback to INIT_PRICE_2020 ---
        # Use explicit None checks (avoid `or`, which misbehaves for valid zeros).
        predicted = parcel_chars.get("PREDICTED_PRICE", None)
        init_2020 = parcel_chars.get("INIT_PRICE_2020", None)

        if predicted is not None:
            self.initial_price = _to_float(predicted, "PREDICTED_PRICE")
        elif init_2020 is not None:
            self.initial_price = _to_float(init_2020, "INIT_PRICE_2020")
        else:
            raise KeyError(
                "Parcel is missing initial price. Provide 'PREDICTED_PRICE' or 'INIT_PRICE_2020'."
            )

        self.last_transaction_price = float(self.initial_price)

        # Initialize market state
        self.ask_price = None
        self.market_price = None

        # Make ownership explicit (Model sets this later)
        self.owner = None

        # --- Clean and store required hedonic attributes (typed) ---
        # These are the features used by the hedonic model; keep strict.
        self.AGE = _to_float(parcel_chars.get("AGE"), "AGE")
        self.HOUSESIZE = _to_float(parcel_chars.get("HOUSESIZE"), "HOUSESIZE")
        self.LOTSIZE = _to_float(parcel_chars.get("LOTSIZE"), "LOTSIZE")
        self.ROOMS = _to_int(parcel_chars.get("ROOMS"), "ROOMS")
        self.QUALITY = _to_float(parcel_chars.get("QUALITY"), "QUALITY")
        self.LN_DIST_CBD = _to_float(parcel_chars.get("LN_DIST_CBD"), "LN_DIST_CBD")
        self.LN_DIST_MEUSE = _to_float(parcel_chars.get("LN_DIST_MEUSE"), "LN_DIST_MEUSE")

        # Optional raw distance (can be helpful for amenity/risk logic elsewhere)
        dist_meuse_raw = parcel_chars.get("DIST_MEUSE", None)
        self.DIST_MEUSE = None if dist_meuse_raw is None else _to_float(dist_meuse_raw, "DIST_MEUSE")

        # --- Flood dummy (STRICT: FP_PROTECTED REQUIRED) ---
        # Flood status is a core treatment variable in the CEUS paper.
        # We require it to be present and strictly coded as:
        #   - bool: True/False
        #   - numeric: 1/0
        #
        # Any missing value, None, or non-binary coding raises an error.

        if "FP_PROTECTED" not in parcel_chars:
            raise KeyError(
                f"Parcel {self.unique_id}: Missing required flood-prone indicator 'FP_PROTECTED'."
            )

        flood_val = parcel_chars["FP_PROTECTED"]

        if flood_val is None:
            raise ValueError(
                f"Parcel {self.unique_id}: 'FP_PROTECTED' cannot be None."
            )

        if isinstance(flood_val, bool):
            self.d_floodprone = int(flood_val)

        elif isinstance(flood_val, (int, float)):
            if flood_val in (0, 1):
                self.d_floodprone = int(flood_val)
            else:
                raise ValueError(
                    f"Parcel {self.unique_id}: 'FP_PROTECTED' must be 0 or 1, got {flood_val}."
                )

        else:
            raise TypeError(
                f"Parcel {self.unique_id}: 'FP_PROTECTED' must be bool or 0/1 numeric, got {type(flood_val)}."
            )


    # ------------------------------------------------------------------
    # Convenience properties / setters
    # ------------------------------------------------------------------

    @property
    def current_price(self) -> float:
        """Safe realized-price fallback (always the last realized transaction price)."""
        return float(self.last_transaction_price)
    
    def set_market_price(self, price: float) -> None:
        """Set the market price (e.g., realtor estimate)."""
        self.market_price = float(price)

    def set_ask_price(self, price: float) -> None:
        """Set the ask/listing price (seller decision)."""
        self.ask_price = float(price)

    def record_sale(self, transaction_price: float) -> None:
        """
        Record a completed sale.
        (Ownership transfer is typically handled elsewhere; this just stores the realized price.)
        """
        self.last_transaction_price = float(transaction_price)
        # After a sale, you can decide whether to clear ask_price/market_price; leaving them
        # can be helpful for debugging. If you prefer, uncomment:
        # self.ask_price = None
        # self.market_price = None

    def get_prop_chars(self) -> Dict[str, float]:
        """
        Return validated parcel characteristics for the hedonic regression.

        Output is a dict with a stable set of keys and numeric values.
        If any required key is missing or not numeric, we raise an informative error.
        """
        chars: Dict[str, float] = {
            "AGE": float(self.AGE),
            "HOUSESIZE": float(self.HOUSESIZE),
            "LOTSIZE": float(self.LOTSIZE),
            "ROOMS": float(self.ROOMS),  # regression expects numeric; keep float for consistency
            "QUALITY": float(self.QUALITY),
            "LN_DIST_CBD": float(self.LN_DIST_CBD),
            "LN_DIST_MEUSE": float(self.LN_DIST_MEUSE),
            "d_floodprone": float(self.d_floodprone),
        }

        # Final sanity check: stable order, all finite numbers
        for key in self._FEATURE_ORDER:
            if key not in chars:
                raise KeyError(f"Parcel feature '{key}' missing in get_prop_chars().")
            val = chars[key]
            if not isinstance(val, (int, float)):
                raise TypeError(f"Parcel feature '{key}' must be numeric, got {type(val)}.")
        return chars

    def __repr__(self) -> str:
        return (
            f"Parcel(id={self.unique_id}, price={self.current_price:.2f}, "
            f"fp={self.d_floodprone}, age={self.AGE}, size={self.HOUSESIZE})"
        )

