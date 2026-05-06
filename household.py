# -*- coding: utf-8 -*-

"""

@author: asli-mutlu

Household class of the RHEA-NL model, based on the Agent class of the MESA library.
This class contains the "stages" for the MESA StagedActivation scheduler for
a household agent.

The Household class contains the following helper functions:
    - set_income()          : Draw household income from given income distribution

    For buyers:
    - compute_utility()     : Compute utility for given properties
    - get_affordable()      : Select properties within household budget range
    - find_best_props()     : Select best properties to bid on
    - place_bid()           : Determine bid price from property ask price

    For sellers:
    - negotiation()         : Trade negotiation process
    - transfer_property()   : Change property ownership from seller to buyer

"""

import numpy as np
import logging

from scipy.stats import truncnorm
from mesa import Agent

class Household(Agent):
    """Household agent class of the RHEA model. """

    def __init__(self, model, income, F_income, income_group, price_margin_bounds):
        """Initialize a household agent.

        Args:
            model (Model)              : Reference to the model instance
            income (float)             : Annual household income
            F_income (float)           : Fraction of income spent on housing
            income_group (int)         : Income group (1 = lowest)
            price_margin_bounds (tuple): Range for base bid/ask margin (±%)
        """


        super().__init__(model.next_id(), model)

        # -------------------- Financial Attributes -------------------- #
        self.income = income
        self.F_income = F_income
        self.income_group = income_group
        self.budget = int(F_income * income * model.mortgage_years)
        self.search_budget = 0  # Will be updated later

        if price_margin_bounds == (0.0, 0.0):
            self.base_price_margin = 0.0
        else:
            self.base_price_margin = np.round(model.rng.uniform(*price_margin_bounds), 3)

        self.price_margin = self.base_price_margin

        # -------------------- Market & Property -------------------- #
        self.property = None
        self.ask_price = None
        self.market_status = "Inactive"
        self.was_seller = False
        self.was_buyer = False
        self.n_tr_nosuccess = 0  # Count of unsuccessful transactions
        self.moving_wait_time = 0
        self.bids_placed = {}

        # ---------------- Model-Wide Constants (Accessed) ---------------- #
        self.listing_time_discount = self.model.listing_time_discount
        self.flood_damage = self.model.flood_damage
        # self.decay_rate = self.model.decay_rate
        self.flood_probability_obj = self.model.flood_probability_obj
        self.min_acceptable_bid_ratio = self.model.min_acceptable_bid_ratio

        # -------------------- Residence Time -------------------- #
        mean, std = self.model.res_time_years_avg, self.model.res_time_years_sd
        min_years = self.model.not_moving_years
        max_years = self.model.years

        # Sample and clip to acceptable range
        years = np.round(self.model.rng_init.normal(mean, std))
        self.res_time_years = min(max(years, min_years), max_years)
        self.res_time_steps = self.res_time_years * self.model.kY

        # Set purchase step (backdate at initialization)
        if self.model.schedule.steps == 0:
            self.purchase_step = -np.round(self.model.rng.uniform(0, self.res_time_steps))
        else:
            self.purchase_step = self.model.schedule.steps

        # -------------------- NbS Preferences -------------------- #
        # Model must define: alpha_nbs, beta_nbs, cv_nbs, sigma_nbs (sigma from CV)
        # Store alpha/beta locally for faster access and draw a permanent taste multiplier η_i.
        self.alpha_nbs = getattr(self.model, "alpha_nbs", 0.0)
        self.beta_nbs  = getattr(self.model, "beta_nbs",  0.0)
        self.sigma_nbs = getattr(self.model, "sigma_nbs", 0.0)  # e.g., 0.25
        self.gamma_flood = getattr(self.model, "gamma_flood", self.beta_nbs)  # default to β_nbs if not provided

        # Household-specific taste multiplier η_i ~ LogNormal(0, sigma^2)
        # mean(η_i) = exp(0.5*sigma^2); median(η_i)=1.0
        if self.sigma_nbs > 0:
            self.nbs_taste = float(np.exp(self.model.rng_init.normal(0.0, self.sigma_nbs)))
        else:
            self.nbs_taste = 1.0  # neutral if sigma==0

        # -------------------- Risk Perception (RP BIAS) -------------------- #
        if self.model.use_RP_bias:
            if self.model.RP_bias_distribution == "Normal":
                self.RP_bias = self.model.rng_init.normal(*self.model.RP_bias_normal)
            elif self.model.RP_bias_distribution == "Beta":
                self.RP_bias = self.model.rng_init.beta(*self.model.RP_bias_beta)
            else:
                raise ValueError("Invalid RP_bias_distribution. Choose 'Normal', 'Beta', or 'Empirical'.")
        else:
            self.RP_bias = 0

        # Remove hard clipping to [0,1] because empirical bias can be >1 or <0
        # Keep NaN-safe guard
        if self.RP_bias is None or np.isnan(self.RP_bias):
            self.RP_bias = 0.0

        # -------------------- Perceived Flood Probability -------------------- #
        # self.set_perceived_FP()  # Based on RP bias and residence time
        self.get_restime_obj_FP()


    def _parcel_key(self, p):
        """Stable key for parcel mapping; avoids using Parcel objects as dict keys."""
        uid = getattr(p, "unique_id", None)
        return int(uid) if uid is not None else id(p)


    def update_search_budget(self):
        """Update household's search budget based on their price margin."""
        
        # Ensure price_margin is defined, raise error if not
        if not hasattr(self, "price_margin"):
            raise ValueError(f"Household {self.unique_id} does not have a 'price_margin' attribute.")

        # Calculate new search budget based on the maximum price margin
        self.search_budget = np.round((self.budget / (1 + self.price_margin)), -2)


    def get_affordable(self, properties, lower_limit=True):
        """Get affordable properties based on household search budget.

        Args:
            properties (list): List of available properties
            lower_limit (Boolean): If TRUE, select properties above lower limit
        Returns:
            props_afford_list (list): List of properties within search budget range
        """

        # Raise ValueError if search budget is not correctly defined
        if self.search_budget == 0:
            raise ValueError(
                f"Search budget is zero for household {self.unique_id} | "
                f"Market status: {self.market_status} | "
                f"Income: {self.income} | "
                f"Initial budget: {self.budget}"
            )
        
        # Check if ask price determined correctly
        for prop in properties:
            if prop.ask_price is None:
                raise ValueError("Ask price not provided. Ensure seller sets the ask price correctly.")

        if lower_limit:
            # Filter properties based on effective price and budget
            props_afford = filter(lambda prop: self.model.lower_bid_limit <= prop.ask_price / self.search_budget <= 1, properties)

        else:
            # Compare property prices only to household search budget
            props_afford = filter(lambda prop: prop.ask_price <= self.search_budget, properties)

        # Convert filtered objects to list
        props_afford_list = list(props_afford)
        
        return props_afford_list
    
    def get_restime_obj_FP(self):
        """Calculate buyer household's perceived flood probability over their residence time."""
        p_flood = self.model.flood_probability_obj
        if not (0 <= p_flood <= 1):
            raise ValueError("Flood probability scenario must be between 0 and 1.")
        if self.res_time_years <= 0:
            raise ValueError("Residence time must be positive.")

        # Objective flood probability over residence time (no bias)
        self.restime_obj_FP = np.round(np.clip((1 - (1 - p_flood) ** self.res_time_years), 0, 1), 3)

        self.objective_FP = self.restime_obj_FP

     
    def compute_utility(self, properties, method = None):
        """Compute expected utility for properties (continuous NbS amenities + heterogeneous tastes).

        EU_ij = P_j                              # base price (ask or market)
                + P_j * a_i(d_nbs)               # amenity add-on
                - π_i(d_flood) * L * P_j         # expected flood loss on base price
        then apply listing-time discount at the end.

        Notes:
        - a_i(d_river) = η_i * α * exp(-β d)          # individual amenity taste (η_i) * mean curve (α * exp(-β d))
        - π_i(d_river) = perceived_FP (uniform) or perceived_FP * exp(-gamma_flood * d_flood) if distance_based
        - Flood loss uses the un-amenitized base price P_j to avoid double counting.
        """

        method = method or self.model.buyer_util_method
        allowed = {"EU_v1_Dutch", "EU_v1_Dutch_Env"}
        if method not in allowed:
            raise ValueError(f"Invalid buyer_util_method='{method}'. Choose from {sorted(allowed)}.")

        # Base price P_j: prefer ask; if missing, use Realtor's market price
        base_price = np.array([
            (prop.ask_price if getattr(prop, "ask_price", None) not in (None, 0) else getattr(prop, "market_price", 0.0))
            for prop in properties
        ], dtype=float)

        # Flood-prone indicator as stored on the Parcel
        d_floodprone = np.array([getattr(prop, "d_floodprone", 0) for prop in properties], dtype=float)
        d_river = np.array([getattr(prop, "DIST_MEUSE", 0) for prop in properties], dtype=float)

        # Listing time (for discount)
        h_listing_time = np.array([getattr(prop, "listing_time", 0) for prop in properties], dtype=float)

        # --- Amenity term: a_i(d_river) = η_i * alpha * exp(-beta * d) ---
        if method == "EU_v1_Dutch_Env" and (self.alpha_nbs > 0) and (self.beta_nbs > 0):
            mu_d = self.alpha_nbs * np.exp(-self.beta_nbs * d_river)  # mean curve
            amenity_factor = self.nbs_taste * mu_d                  # a_i(d)
        else:
            amenity_factor = np.zeros_like(base_price)

        amenity_value = base_price * amenity_factor  # P * a_i(d)
        
        # Start from base (clipped to avoid 0/1 extremes)
        eps = 1e-9 
        base = np.clip(self.restime_obj_FP, eps, 1.0 - eps)

        if self.model.perception_space == "logit":
            # logit transform
            logit_val = np.log(base / (1.0 - base))

            # # global intensity multiplier used for Local Sensitivity Analysis (default 1.0)
            # k = float(getattr(self.model, "bias_intensity_scale", 1.0))

            k_vis_raw = getattr(self.model, "bias_k_visibility", None)
            k_vis = float(1.0 if k_vis_raw is None else k_vis_raw)

            k_soc_raw = getattr(self.model, "bias_k_socio", None)
            k_soc = float(1.0 if k_soc_raw is None else k_soc_raw)

            # keep original for debugging deltas
            _logit0 = logit_val.copy()

            # apply distance decay if requested
            if self.model.flood_risk_mode == "distance_based":
                logit_val -= k_vis * (self.model.gamma_flood * d_river)
            
            # apply socio-institutional bias if requested
            if self.model.use_RP_bias and (self.model.bias_channel == "additive_logit"):
                logit_val -= k_soc * self.model.gamma_bias * float(self.RP_bias)
            
            prop_flood_prob = d_floodprone * (1.0 / (1.0 + np.exp(-logit_val)))

        else:
            # probability-space implementation
            prop_flood_prob = self.restime_obj_FP
            if self.model.flood_risk_mode == "distance_based":
                prop_flood_prob *= np.exp(-self.model.gamma_flood * d_river)
            prop_flood_prob *= d_floodprone

        # clip to [0,1]
        prop_flood_prob = np.clip(prop_flood_prob, 0.0, 1.0)

        # Expected flood loss on the UN-amenitized base price
        expected_flood_loss = prop_flood_prob * self.flood_damage * base_price

        # --- Expected Utility ---
        EU = base_price + amenity_value - expected_flood_loss

        # Store flood loss for transactions/debug
        for i, prop in enumerate(properties):
            prop.expected_flood_loss = float(expected_flood_loss[i])
            prop.amenity_value = float(amenity_value[i])

        # Listing-time discount applied at the end
        cap = getattr(self.model, "listing_discount_cap", 0.15)  # default 15%
        listing_discount = np.clip(np.log1p(h_listing_time) * self.listing_time_discount, 0.0, cap)
        EU *= (1.0 - listing_discount)

        # Round to the nearest hundred for stability
        utilities_dict = {self._parcel_key(prop): float(np.round(EU[i], -2))
                  for i, prop in enumerate(properties)}
        return utilities_dict

    def find_best_props(self, properties):
        """From specified list, find and return the best properties based on utilities.

        Returns:
            dict: {parcel_key: utility} sorted by descending utility, limited to max_bid_sample.
        """
        if properties is None or len(properties) == 0:
            return {}

        utilities = self.compute_utility(properties, self.model.buyer_util_method)  # {key: utility}
        if not utilities:
            return {}

        max_bid_sample = 5
        min_fit = 0.0

        keys = list(utilities.keys())
        weights = np.array([utilities[k] for k in keys], dtype=np.float64)

        weights[~np.isfinite(weights)] = 0.0
        weights[weights < min_fit] = 0.0

        total = float(np.sum(weights))
        if total <= 0.0:
            return {}

        probs = weights / total
        n_pos = int(np.sum(probs > 0))
        k = min(max_bid_sample, len(keys), n_pos)
        if k <= 0:
            return {}

        chosen_keys = self.model.rng.choice(keys, size=k, replace=False, p=probs)

        chosen = {int(key): float(utilities[int(key)]) for key in chosen_keys}
        chosen_sorted = dict(sorted(chosen.items(), key=lambda item: item[1], reverse=True))
        return chosen_sorted

    def set_bid_margin(self):
        """Adjust bid margins based on past failures."""
    
        self.price_margin = self.base_price_margin  # Store previous margin for logging

        # Adjust based on unsuccessful trade attempts (desperation factor)
        if self.n_tr_nosuccess > 0:
            decay_rate = 0.5
            failure_adjustment = (1 - np.exp(-decay_rate * self.n_tr_nosuccess)) * self.model.failure_adjustment
            self.price_margin += failure_adjustment

        # Ensure bid margin stays within limits
        self.price_margin = np.clip(self.price_margin, -0.10, 0.10)  # -10% to +10%

    def place_bid(self, utility, normalized_utility):
        """ Determine the bid price based on the property's utility and past failures.

        Args:
            utility (float): The computed utility value of the property.
            normalized_utility (float): Normalized utility score (0 to 1) for adjusting bid margins.

        Returns:
            bid (float): The final bid price.
        """

        # Get the global bid margin set by set_bid_margin()
        bid_margin = self.price_margin 

        # Adjust for property-specific (local) effects.
        # Calculate the price offset based on the normalized utility
        price_offset = normalized_utility
        if bid_margin > 0:
            adjusted_margin = bid_margin * price_offset
        
        else:
            adjusted_margin = bid_margin * (2 - price_offset)
        
        # Place bid based on the adjusted margin
        bid = np.round((utility * (1 + adjusted_margin)), -2)
        return bid
    
    def remaining_mortgage(self):
        """
        Calculate the remaining mortgage debt after the household's residence time,
        accounting for Loan-to-Value (LTV) ratio.
        Supports both standard and zero-interest mortgage calculations.
        """

        # Get the purchase price of the property
        purchase_price = np.round(self.property.last_transaction_price, -2)

        # Apply Loan-to-Value ratio (default = 1.0 if not set)
        LTV = getattr(self, "LTV", 1.0)
        loan_amount = purchase_price * LTV
        remaining_debt = loan_amount

        # Loan parameters in simulation terms
        n_steps = self.model.mortgage_years * self.model.kY  # total repayment steps
        r = self.model.mortgage_interest_per_step            # interest per step

        # Annuity repayment per step
        if r > 0:
            payment_per_step = remaining_debt * (r * (1 + r) ** n_steps) / ((1 + r) ** n_steps - 1)
        else:
            payment_per_step = remaining_debt / n_steps  # linear repayment

        # How many steps since purchase
        res_time_steps = self.model.schedule.steps - self.purchase_step

        # Remaining debt after res_time_steps
        if res_time_steps > 0:
            if r > 0:
                remaining_debt = remaining_debt * ((1 + r) ** res_time_steps) - (
                    (payment_per_step * ((1 + r) ** res_time_steps - 1)) / r
                )
            else:
                remaining_debt = np.round(max(0, remaining_debt - (payment_per_step * res_time_steps)), -2)

        return max(0, remaining_debt)

    def seller_min_acceptable_bid(self):
        """Determine the minimum price a seller is willing to accept."""
        if self.model.use_mortgage_debt:
            self.min_acceptable_bid = self.remaining_mortgage()
        else:
            self.min_acceptable_bid = np.round(self.property.last_transaction_price * self.min_acceptable_bid_ratio, -2)
        
        return self.min_acceptable_bid

    def adjust_pricing_margin(self):
        """Adjust ask price and negotiation flexibility based on past failures.
        Args:
            market_price (float): The estimated market price from the realtor.

        Returns:
            float: The adjusted ask price.
        """
        self.price_margin = self.base_price_margin  # Store for logging

        # Adjust based on unsuccessful trade attempts
        if self.n_tr_nosuccess > 0:
            decay_rate = 0.5
            failure_adjustment = (1 - np.exp(-decay_rate * self.n_tr_nosuccess)) * self.model.failure_adjustment
            self.price_margin -= failure_adjustment

    def find_highest_bidder(self):
        # Get highest bid and bidder, register others as unsuccessful
        self.highest_bidder = max(self.bids_received,
                                key=self.bids_received.get)
        self.highest_bid = self.bids_received.pop(self.highest_bidder)
        return (self.highest_bidder, self.highest_bid)
    

    def negotiation(self, bids, highest_bid):

        """Ensure property is sold only once and only to bidders who meet or exceed the ask price."""

        bids_sorted = dict(sorted(bids.items(), key=lambda x: x[1], reverse=True))

        if not bids_sorted:
            # If no bids are received, increment the unsuccessful trade count
            self.n_tr_nosuccess += 1
            return

        property_sold = False

        # Build a resolver from parcel_key -> Parcel object
        # Using current market list is usually enough because desired props are drawn from it.
        id_to_prop = {self._parcel_key(p): p for p in self.model.props_for_sale}

        this_key = self._parcel_key(self.property) if self.property is not None else None

        for buyer, bid in bids_sorted.items():

            if buyer.property is not None or self.property is None:
                continue

            if bid < self.negotiation_threshold:
                break  # bids sorted desc; remaining bids are lower

            # Check if buyer has a higher preferred property where they are highest bidder
            buyer_desired = getattr(buyer, "desired_props", {}) or {}
            u_this = buyer_desired.get(this_key, None)
            if u_this is None:
                u_this = -np.inf  # treat as very low preference if missing

            redirected = False

            for alt_key, u_alt in buyer_desired.items():
                if u_alt <= u_this:
                    continue

                alt_prop = id_to_prop.get(int(alt_key))
                if alt_prop is None or alt_prop.owner is None:
                    continue

                if alt_prop.owner.market_status == "Seller" and alt_prop.owner.bids_received:
                    highest_bidder, alt_highest_bid = alt_prop.owner.find_highest_bidder()

                    if highest_bidder is buyer:
                        alt_prop.owner.register_transaction(buyer, alt_highest_bid, alt_highest_bid)
                        alt_prop.owner.n_tr_nosuccess = 0
                        redirected = True
                        property_sold = True
                        break

            if redirected:
                break

            # Otherwise sell THIS property to this buyer
            self.register_transaction(buyer, bid, highest_bid)
            self.n_tr_nosuccess = 0
            property_sold = True
            break

        if property_sold:
            # Successful attempt for seller
            self.n_tr_nosuccess = 0
        else:
            # Unsuccessful attempt for seller
            self.n_tr_nosuccess += 1


    def register_transaction(self, buyer, bid, highest_bid):
        self.model.register_transaction(self.property,
                                    self.unique_id,
                                    buyer.unique_id,
                                    self.ask_price,
                                    bid,
                                    bid,
                                    highest_bid)
        self.transfer_property(buyer, self.property)

    def transfer_property(self, buyer, prop):
        """Transfer a property from the current owner (self) to a buyer.

        Args:
            buyer (Household)   : Buyer to transfer property to
            prop (Parcel)       : Parcel object to transfer
        """

        # Transfer house to new owner
        prop.N_sales += 1
        prop.owner = buyer
        prop.ask_price = None
        buyer.property = prop
        self.property = None
        self.n_tr_nosuccess = 0 
        self.ask_price = None

        # Reset buyers trade attributes and change market status
        buyer.desired_props.clear() 
        buyer.bids_placed = {}

        buyer.n_tr_nosuccess = 0
        buyer.market_status = "Inactive"
        buyer.was_buyer = True
        buyer.moving_wait_time = self.model.not_moving_steps
        buyer.purchase_step = self.model.schedule.steps

        # Sellers leave town with 70% probability, otherwise become buyers
        if self.model.rng.random() < self.model.F_leaving:
            self.model.remove_household(self)
        else:
            self.market_status = "Buyer"
            self.was_seller = True
            # If seller stays: reset seller trade attributes
            self.n_tr_nosuccess = 0
            self.ask_price = None
            self.bids_received = {}


    def stage0(self):
        pass

    def stage1(self):
        """First stage of household step
        All households (Inactive):
            1) Decrease moving wait time 
        Sellers: 
            2) Determine minimum acceptable ask price (fixed threshold (%) or remaining mortgage)
            3) Get market information and adjust negotiation margins
            4) Set ask price based on Realtor's market price and agent's negotiation margins
            5) Update properties listing time
            6) Initialize bid dictionary
        """
        
        # Reset was_seller flag at the start of a new step
        self.was_seller = False
        # Check if household has recently moved and has to wait for new trade
        if self.moving_wait_time > 0:
            self.moving_wait_time -= 1

        if self.market_status == "Seller":
            self.seller_min_acceptable_bid()
            
            # Adjust price margins for negoatiation dynamically based on previous failures
            self.adjust_pricing_margin()

            # Determine the ask price based on market expectations
            if self.price_margin == 0.0:
                self.ask_price = self.property.market_price
                self.negotiation_threshold = self.ask_price
            elif self.price_margin > 0:  
                # Seller has positive market expectations or no urgency to sell
                self.ask_price = max(np.round(self.property.market_price * (1 + self.price_margin), -2), self.min_acceptable_bid)
                self.negotiation_threshold = np.round(max(self.property.market_price, self.min_acceptable_bid), -2)
            else:  
                # Seller has negative market expectations or urgency to sell
                self.ask_price = max(self.property.market_price, self.min_acceptable_bid)
                self.negotiation_threshold = np.round(self.ask_price * (1 + self.price_margin), -2) 

            # Assign ask price to the property
            self.property.ask_price = self.ask_price

            # Update the listing time based on number of unsuccessful transactions
            self.property.listing_time = self.n_tr_nosuccess

            # Initialize bid dictionary
            self.bids_received = {}

    def stage2(self):
        """Second stage of household step:
        Buyers:
            1) Update bid margin
            2) Update search budget
            3) Select affordable properties
            4) Select desired property (highest utility) from subset of
                available properties
            5) Place bid
        """
        
        if self.market_status == "Buyer":
            # Update bid margin based on market and past trading experience
            self.set_bid_margin()

            # Update search budget
            self.update_search_budget()

            # Get affordable properties within budget
            props = self.get_affordable(self.model.props_for_sale)
                
            if props is None or len(props) == 0:
                    self.desired_props = {}
                    return

            # sample subset
            if len(props) > self.model.market_subset:
                props = self.model.rng.choice(props, self.model.market_subset, replace=False)

            # map keys -> Parcel objects for later owner access
            id_to_prop = {self._parcel_key(p): p for p in props}

            desired = self.find_best_props(props)   # dict {parcel_key: utility}
            self.desired_props = desired

            if not desired:
                return

            # normalize utilities over desired set
            utils = np.array(list(desired.values()), dtype=float)
            utils[~np.isfinite(utils)] = np.nan
            if not np.any(np.isfinite(utils)):
                self.desired_props = {}
                return

            min_u = float(np.nanmin(utils))
            max_u = float(np.nanmax(utils))
            denom = max_u - min_u

            for key, u in desired.items():
                if not np.isfinite(u):
                    continue

                if denom <= 0.0 or not np.isfinite(denom):
                    u_norm = 1.0
                else:
                    u_norm = (float(u) - min_u) / denom
                    u_norm = float(np.clip(u_norm, 0.0, 1.0))

                prop = id_to_prop.get(int(key))
                if prop is None or prop.owner is None:
                    continue

                # IMPORTANT: If bid base should be a price (not the EU value), 
                # use ask_price if available, else market_price.
                # Here, we use EU as EU already includes price
                # base_price = float(getattr(prop, "ask_price", None) or getattr(prop, "market_price", 0.0))
                # if not np.isfinite(base_price) or base_price <= 0:
                #     continue

                # bid = float(self.place_bid(utility=base_price, normalized_utility=u_norm))
                
                bid = float(self.place_bid(utility=float(u), normalized_utility=u_norm))
                prop.owner.bids_received[self] = bid

    def stage3(self):
        """Third stage of household step:
            1) Check if any bids received (seller)
            2) Select highest bid (seller)
            3) Start negotiation process (seller --> buyer)
        """

        if self.market_status == "Seller":
            # Check if seller received any bids
            if self.bids_received:
                self.find_highest_bidder()

                # Start negotiation between seller and highest bidder
                self.negotiation(self.bids_received, self.highest_bid)

            else:
                self.n_tr_nosuccess += 1
               
    def stage4(self):
        """Fourth stage of household step:
            1) If unsuccessful search > time limit, leave area (buyers)
            2) If unsuccessful sales > time limit, give up (sellers)
        """
        
        # Increment number of unsuccessful events for unsuccessful buyers, ensure Buyer was not a seller
        if self.market_status == "Buyer" and self.property is None and self.was_seller is False:
            self.n_tr_nosuccess += 1
           
        buyer_limit = self.model.buyer_timelimit * self.model.kY
        seller_limit = self.model.seller_timelimit * self.model.kY

        # Buyers leave market after predefined timelimit
        if self.market_status == "Buyer" and self.n_tr_nosuccess >= buyer_limit:
            self.model.remove_household(self)

        # Sellers give up on selling after predefined timelimit
        elif (self.market_status == "Seller" and
              self.n_tr_nosuccess >= seller_limit):
            self.market_status = "Inactive"
            self.n_tr_nosuccess = 0
            self.property.ask_price = None
            self.moving_wait_time = self.model.not_moving_steps
        elif self.market_status == "Inactive":
            if self.property is not None:
                self.property.ask_price = None
