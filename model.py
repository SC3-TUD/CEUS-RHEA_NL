# -*- coding: utf-8 -*-

"""

@author: asli-mutlu

Model class for RHEA-NL housing market model tailored to the Netherlands.

The RHEA-NL model simulates the aggregated impact of household residential
location choices under natural hazard risks. The model consists of realtor
and household agents forming ask and bid prices from adaptive price
expectations. Households are heterogeneous in income, risk perceptions and
preferences for nature-based amenities.

The implementation of the RHEA-NL model is based on the Python Mesa framework
for agent-based modeling (https://mesa.readthedocs.io/en/stable/) and is
written in Python version 3.10. 

"""

import numpy as np
import pandas as pd
import math
import logging

from mesa import Model
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector
from household import Household
from parcel import Parcel
from realtor import Realtor

# # Initialize logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

MORTGAGE_YEARS = 30             # Mortgage duration (years)
ANNUAL_MORTGAGE_INTEREST = 0    # Realistic values for the Dutch market: 0.03; 0.04; 0.05. Use 0 for linear repayment.
NEW_BUYER_COEF = 0.7            # Additional buyer/seller ratio parameter
F_INCOME_BY_GROUP = {           # 1 = the lowest income group, 5 = the highest income group
    1: 0.30,
    2: 0.275,
    3: 0.25,
    4: 0.225,
    5: 0.20
}
F_LEAVING = 0.7                 # Fraction of sellers leaving area after sale
F_FLOOD_DAMAGE = 0.17           # Percentage of house value damaged in case of flood
MARKET_SUBSET = 10              # Subset size of properties seen by buyers 
BUYER_TIMELIMIT = 2             # Max time on the market for buyers (in years)
SELLER_TIMELIMIT = 2            # Max time on the market for sellers
NOT_MOVING_YEARS = 2            # Wait time after recently moving (in years) # ASLI: original was 0.5
YEARS_TRANS_HIST = 2            # Repetitive parameter - Realtor 
RES_TIME_YEARS = (17, 1.5)      # Years of average residence time and standard deviation
N_INCOME_GROUPS = 5             # Change this to 10 for deciles, 5 for quintiles, etc.
INCOME_GROUP_FRACTION = 1.0 / N_INCOME_GROUPS  # Automatically calculates the fraction per group
MAX_BID_SAMPLE = 5
LISTING_DISCOUNT_CAP = 0.15
# NbS preference curve (continuous) and heterogeneity
ALPHA_NBS = 0.02066136          # Bockarjova et al. (2020)
BETA_NBS  = 0.00153330          # Bockarjova et al. (2020)
SIGMA_NBS = 0.25                # Standard deviation of the NbS preference heterogeneity derived from CV = 0.26
FLOOD_HALFLIFE_M = 1000         # Half-life for flood risk decay (meters); set to 0.0 for uniform risk perception
GAMMA_BIAS = 1.70               # gamma_bias scales the additive logit bias term.
                                # Calibrated so visibility decay and other distortion each explain ~50% 
                                # of the average perception gap (based on mean_d (distance to river) and 
                                # mean_b (bias in a representative household population)).
SIGMA_FLOOD = 0.25
GAMMA_CD_AMEN = 0.50            # Amenity exponent (water)
ETA_WATER = 0.062               # Water amenity decay; pick to match ln(dist_meuse) elasticity via gamma*eta
TAU_CBD = 3000.0                # €/km/year generalized disutility of CBD distance



# ------------------------------------------------ #
# ----------- DATACOLLECTION FUNCTIONS ----------- #
# ------------------------------------------------ #
def count_by_status(model, market_status):
    """Get agent count by market status ("Buyer", Seller" or "Inactive")."""
    agent_count = sum(1 for hh in model.households
                      if hh.market_status == market_status)
    return agent_count

class RHEA_Model(Model):
    """Model class for the RHEA model. 

    Default Model Configuration:
    ----------------------------
    These defaults represent a baseline equilibrium test case with:

    - No flood risk                    --> (flood_probability_obj=0.00)
    - No risk perception bias          --> (use_RP_bias=False)
    - No listing time penalty          --> (listing_time_discount=0.0)
    - No bidding or pricing stragety   --> (price_margin_bounds=(-0.0, 0.0), failure_adjustment=0.0)
    - Balanced demand-supply           --> (buyer_demand = 1)

    To test realistic or experimental settings, use scenario files via `batch_runner.py`.
    """

    def __init__(self, random_seed, parcel_file,
             kY=2, years=30,
             flood_probability_obj=0.0,
             update_hedonics=True, dynamic_income_groups=False,
             buyer_util_method="EU_v1_Dutch",
             balance_sellers=False,
             seller_mode="Probabilistic", NbS_mode="distance_based",
             NbS_prefs_uniform=(0.045, 0.055),
             use_RP_bias=False,
             use_uniform_res_time=True, use_mortgage_debt=False,
             flood_damage=0.17,
             decay_rate=0.001,
             flood_risk_mode="distance_based",
             lower_bid_limit=0.75, min_acceptable_bid_ratio=0.70,
             listing_time_discount=0.0,
             RP_bias_distribution="Normal",
             RP_bias_normal=(0.0, 0.5), RP_bias_beta=(4, 1.5),
             buyer_demand=1.0,
             price_margin_bounds=(-0.0, 0.0),
             failure_adjustment=0.0, max_failure_adjustment=0.05,
             empirical_bias_dict=None,
             perception_space="logit",       # "prob" or "logit"
             bias_channel="none",           # "none" (S1c) or "additive_logit" (S1d)
             bias_intensity_scale: float | None = None,
             bias_k_visibility: float | None = None,           
             bias_k_socio: float | None = None,
             # --- formerly hard-coded defaults; can be overridden per scenario / GSA ---
             gamma_bias: float | None = None,
             alpha_nbs: float | None = None,
             beta_nbs: float | None = None,
             sigma_nbs: float | None = None,
             flood_halflife_m: float | None = None,
             ):
    
        
        super().__init__()
        logging.info("Initializing RHEA Model...")
        """Initialization of the RHEA model.

        Args:
            random_seed (int):                  Seed value for consistent random number generation.
            parcel_file (str):                  Path to the CSV file containing initial parcel data.
            kY (int):                           Number of timesteps per year.
            years (int):                        Number of years to simulate (total timesteps = years * kY).
            
            F_seller (tuple):                   Mean and std dev of the fraction of sellers each timestep.
            F_buyer (tuple):                    Mean and std dev of the fraction of buyers each timestep.

            flood_probability_obj (float): Annual flood probability (e.g., 0.02 for 1-in-50-year flood).
            update_hedonics (bool):             If True, update the regression-based price function at each step.
            buyer_util_method (str):            Utility function used by households. Options: "EU_v1_Dutch", "EU_v1_Dutch_Env".
            
            seller_mode (str):                  Mode to assign sellers each step. Options: "Random" or "Probabilistic".
            NbS_mode (str):                     Distribution of NbS influence. "uniform" for flat effect, "distance_based" for distance-decay effect.
            NbS_prefs_uniform (tuple):          Mean and std dev for household preferences toward NbS (used only in "uniform" mode).

            use_RP_bias (bool):                 Whether households misperceive flood risk.
            use_uniform_res_time (bool):        If True, all households have the same residence time.
            use_mortgage_debt (bool):           Whether sellers consider remaining mortgage debt when setting ask price.

            flood_damage (float):               Share of property value expected to be lost in case of a flood.
            lower_bid_limit (float):            Minimum ratio of bid price to household budget (e.g., 0.75)
            min_acceptable_bid_ratio (float):   Minimum acceptable bid as a percentage of the property's last transaction price (i.e. purchase price of seller).

            listing_time_discount (float):      Percent discount per year that buyers apply to properties with long listing times.

            RP_bias_distribution (str):         "Normal" or "Beta" distribution to model risk perception bias.
            RP_bias_normal (tuple):             Mean and std dev for normally-distributed RP bias.
            RP_bias_beta (tuple):               Alpha and beta parameters for beta-distributed RP bias.

            buyer_demand (float):               Demand pressure factor (e.g., 1.0 = balanced, 1.25 = excess demand).
            price_margin_bounds (tuple):        Range from which each household draws its initial price margin (±% around estimated price).
            
            failure_adjustment (float):              Increment in bid (buyers) or decrement in ask price (sellers) per failed trade attempt.
            max_failure_adjustment (float):          Maximum cumulative adjustment allowed from failed attempts.

    Returns:
        RHEA_Model: An initialized agent-based housing market model.
            
        """

        # -- SCHEDULE INITIALIZATION -- #
        self.current_id = 0
        stage_list = ["stage0", "stage1", "stage2", "stage3", "stage4"]
        self.schedule = StagedActivation(model=self, stage_list=stage_list)
        # IMPORTANT: We rely on the schedule executing agents in insertion order within each stage.
        # Realtor is added first so stage0 sets parcel.market_price before households make decisions.
        # -------------------------------

        # -- REGULATE STOCHASTICITY -- #
        # Separate random generator for household initialization
        self.rng_init = np.random.default_rng(random_seed)
        # Random generator for rest of dynamics
        self.rng = np.random.default_rng(random_seed + 1)
        if seller_mode == "Random":
            # Separate random generator if seller selection is random
            self.rng_random_sellers = np.random.default_rng(random_seed + 2)
        # ------------------------------

        # -- INITIALIZATION -- #

        # Initialize model parameters
        self.kY = kY
        self.years = years
        self.new_buyer_coef = NEW_BUYER_COEF
        self.mortgage_years = MORTGAGE_YEARS
        self.mortgage_interest_per_step = ANNUAL_MORTGAGE_INTEREST / kY
        self.F_leaving = F_LEAVING
        self.market_subset = MARKET_SUBSET
        self.buyer_timelimit = BUYER_TIMELIMIT
        self.seller_timelimit = SELLER_TIMELIMIT
        self.not_moving_years = NOT_MOVING_YEARS
        self.not_moving_steps = np.ceil(NOT_MOVING_YEARS * kY)
        self.max_bid_sample = MAX_BID_SAMPLE
        self.listing_discount_cap = LISTING_DISCOUNT_CAP
        # NbS preference curve (continuous) and heterogeneity (overrideable)
        self.alpha_nbs = ALPHA_NBS if alpha_nbs is None else float(alpha_nbs)
        self.beta_nbs  = BETA_NBS  if beta_nbs  is None else float(beta_nbs)
        self.sigma_nbs = SIGMA_NBS if sigma_nbs is None else float(sigma_nbs)

        # Flood distance-decay half-life (overrideable)
        self.flood_halflife_m = FLOOD_HALFLIFE_M if flood_halflife_m is None else float(flood_halflife_m)
        self.gamma_flood = 0.0 if self.flood_halflife_m <= 0 else (np.log(2.0) / self.flood_halflife_m)


        # Save model setting
        self.balance_sellers = balance_sellers
        self.seller_mode = seller_mode
        self.update_hedonics = update_hedonics
        self.dynamic_income_groups = dynamic_income_groups
        self.dynamic_income_thresholds = None
        self.dynamic_income_bounds = None
        self.buyer_util_method = buyer_util_method
        self.flood_probability_obj = flood_probability_obj
       
        # Market Trend Tracking and Settings
        self.dynamic_income_bounds_history = []
        self.failure_adjustment = failure_adjustment
        self.max_failure_adjustment = max_failure_adjustment
        self.use_RP_bias = use_RP_bias
        self.use_uniform_res_time = use_uniform_res_time
        self.use_mortgage_debt = use_mortgage_debt
        self.RP_bias_distribution = RP_bias_distribution
        self.RP_bias_normal = RP_bias_normal
        self.RP_bias_beta = RP_bias_beta
        self.buyer_demand = buyer_demand
        self.flood_damage = flood_damage
        self.decay_rate = decay_rate
        self.lower_bid_limit = lower_bid_limit
        self.NbS_prefs_uniform = NbS_prefs_uniform
        self.min_acceptable_bid_ratio = min_acceptable_bid_ratio
        self.listing_time_discount = listing_time_discount

        # Cobb-Douglas utility calibration knobs; used only when buyer_util_method=="CD_utility")
        self.gamma_cd_amen = GAMMA_CD_AMEN
        self.eta_water = ETA_WATER
        self.tau1_cbd = TAU_CBD

        # Average residence time in years
        self.res_time_years_avg = RES_TIME_YEARS[0]
        self.res_time_years_sd = RES_TIME_YEARS[1]

        if self.use_uniform_res_time:
            self.res_time_years_sd = 0
        # Average residence time in steps
        self.avg_res_time_steps = self.res_time_years_avg * kY
        self.years_trans_hist = YEARS_TRANS_HIST
        self.n_income_groups =  N_INCOME_GROUPS 
        self.income_group_fraction = INCOME_GROUP_FRACTION 
        self.income_group_data = []  # Stores income group distribution per timestep
        self.price_margin_bounds = price_margin_bounds

        # Call functions to calculate seller_frac and buyer_frac
        self.F_seller = self.proportion_of_sellers()

        # Set risk and NbS mode: "uniform" or "distance_based"
        self.flood_risk_mode = flood_risk_mode
        self.NbS_mode = NbS_mode 

        # Empirical RP-bias distribution (used by households when RP_bias_distribution == "Empirical")
        self.empirical_bias_dict = {} if empirical_bias_dict is None else empirical_bias_dict

        # NEW !!!
        self.perception_space = perception_space
        self.bias_channel = bias_channel


        # Paramaters for sensitivity analysis
        self.bias_intensity_scale = bias_intensity_scale
        self.bias_k_visibility = bias_k_visibility
        self.bias_k_socio = bias_k_socio    

        # ------------------------------------------------ #
        # ------------ AGENT INITIALIZATION -------------- #
        # ------------------------------------------------ #
       
        # # Select case study
        # if use_sample and sample_size:
        #     parcel_file = parcel_file.replace(".csv", f"_sampled_{sample_size}.csv")

        self.parcel_file = parcel_file
        logging.info(f"Loading parcel data from: {parcel_file}")
        self.dataset_name = parcel_file.split("/")[-1].split(".")[0]

        # Initialize lists for parcels and households
        self.parcels = []
        self.households = []

        # Initialize households and parcels
        self.initialize_households(parcel_file)

        # Initialize realtor and transaction history
        self.realtor = Realtor(unique_id="Realtor",model=self)

        # Realtor must be added before households so that Realtor.stage0 runs first
        # and writes parcel.market_price before seller households use it in their stage1.
        self.schedule.add(self.realtor)

        self.transactions = {}

        # --- Compute E[d] from stock ---
        dist_vals = []
        for p in getattr(self, "parcels", []):
            d = getattr(p, "DIST_MEUSE", None)
            if d is None:
                d = getattr(p, "distance_to_river", None)
            if d is not None:
                dist_vals.append(float(d))

        E_d = float(np.mean(dist_vals)) if dist_vals else np.nan

        # --- Compute E[b] from your bias distribution ---
        if self.RP_bias_distribution.lower() == "beta":
            a, b = map(float, self.RP_bias_beta)
            E_b = a / (a + b) if (a + b) != 0 else np.nan
        else:
            mu, sigma = self.RP_bias_normal
            E_b = float(mu)  # Mean of the normal distribution

        # --- Set latent idiosyncratic bias strength (phi / gamma_bias) ---
        use_calibration = (self.use_RP_bias and self.bias_channel == "additive_logit")

        phi_calibrated = None
        if use_calibration:
            # Guard E_b (need finite)
            if (E_b is None) or (E_b == 0) or np.isnan(E_b) or np.isnan(E_d):
                phi_calibrated = None
            else:
                phi_calibrated = self.gamma_flood * (E_d / E_b)

        # Priority:
        # 1) explicit gamma_bias passed in config/scenario
        # 2) calibrated phi (if computable)
        # 3) fallback constant GAMMA_BIAS

        if gamma_bias is not None:
            self.gamma_bias = float(gamma_bias)
        elif (phi_calibrated is not None) and np.isfinite(phi_calibrated):
            self.gamma_bias = float(phi_calibrated)
        else:
            self.gamma_bias = GAMMA_BIAS

    
        # ------------------------------------------------ #
        # -------- INITIALIZE OUTPUT COLLECTION ---------- #
        # ------------------------------------------------ #

        model_reporters = {"Households": lambda m: len(m.households),
                            "N_inactive": lambda m: count_by_status(m, "Inactive"),
                            "N_init_sellers": lambda m: count_by_status(m, "Seller") + len(m.transactions),
                            "N_init_buyers": lambda m: count_by_status(m, "Buyer") - len(m.transactions),
                            "N_Failed_sellers": lambda m: count_by_status(m, "Seller"),
                            "N_Failed_buyers": lambda m: count_by_status(m, "Buyer"), 
                            "N_Successful_transactions": lambda m: len([trans["Seller ID"] 
                                                for trans in m.transactions.values()]),                      
                            "Successful sellers": lambda m: [trans["Seller ID"]
                                                for trans in m.transactions.values()],
                            "Successful buyers": lambda m: [trans["Buyer ID"]
                                                for trans in m.transactions.values()],
                            "Sold properties": lambda m: [tx.get("prop_id") for tx in m.transactions.values()],
                            "P_floodprone": lambda m: [trans["P_floodprone"] for trans
                                                in m.transactions.values()],
                            "P_NbS_distance": lambda m: [trans["P_NbS_distance"] for trans
                                                in m.transactions.values()],
                            "P_ask": lambda m: [trans["P_ask"] for trans
                                                in m.transactions.values()],
                            "P_trans": lambda m: [trans["P_trans"] for trans
                                                in m.transactions.values()],
                            "P_exp_flood_loss": lambda m: [trans["P_exp_flood_loss"] for trans
                                                in m.transactions.values()],
                            "P_amenity_value": lambda m: [trans["P_amenity_value"] for trans
                                                in m.transactions.values()],
                            "P_listing_time": lambda m: [trans["P_listing_time"] for trans
                                                in m.transactions.values()],                            
                            "Trans history": lambda m: m.realtor.last_used_k,
                           }
        if self.update_hedonics:
            model_reporters.update({
                "Realtor model":   lambda m: m.realtor.get_regression_coefs(),            # the dict
                "Realtor R2":      lambda m: float(m.realtor.result.rsquared) if m.realtor.result else None,
                "Realtor nobs":    lambda m: int(m.realtor.result.nobs) if m.realtor.result else 0,
                "Trans history":   lambda m: m.realtor.last_used_k or 0,
            })

        agent_reporters = {# ---------- HOUSEHOLD VARIABLES ------------#
                            "Type": (lambda a: "Household"
                                    if type(a) == Household else "Realtor"),
                            "Market status": (lambda a: a.market_status
                                                if type(a) == Household else None),
                            "Flood probability": (lambda a:a.flood_probability_obj
                                        if type(a) == Household else None),
                            "Income": (lambda a: a.income
                                        if type(a) == Household else None),
                            "Base price margin": (lambda a: a.base_price_margin
                                        if type(a) == Household else None),
                            "Adjusted price margin": (lambda a: a.price_margin
                                        if type(a) == Household else None),
                            "Property ID": (lambda a: a.property.unique_id
                                            if (type(a) == Household) and
                                                (a.property is not None) else None),
                            "Property realtor price": (lambda a: a.property.market_price 
                                                if (type(a) == Household) and
                                                    (a.property is not None) else None),
                            "Property ask price": (lambda a: a.property.ask_price 
                                                if (type(a) == Household) and
                                                    (a.property is not None) else None),
                            "Seller min acceptable bid": (lambda a: a.min_acceptable_bid
                                                if (type(a) == Household) and (a.market_status == "Seller") else None),
                            "Property last transaction price": (lambda a: a.property.last_transaction_price 
                                                if (type(a) == Household) and 
                                                    (a.property is not None) else None),
                            "Property floodprone": (lambda a: int(a.property.d_floodprone)
                                                    if (type(a) == Household) and (a.property is not None) else None),
                            "Property NbS distance": (lambda a: float(a.property.DIST_MEUSE)
                                                    if (type(a) == Household) and (a.property is not None) else 0.0),
                            "NbS preference": (lambda a: getattr(a, "nbs_taste", None)
                                            if type(a) == Household else None),
                            "ResTime Objective FP": (lambda a: a.objective_FP
                                                if (type(a) == Household) else None),
                            "Property N sales": (lambda a: a.property.N_sales
                                                if (type(a) == Household) and
                                                    (a.property is not None) else None),
                            "Agent N Tr No Success": (lambda a: a.n_tr_nosuccess
                                                if type(a) == Household else None),
                            "RP bias": (lambda a: a.RP_bias
                                                if type(a) == Household else None),                                                                        
                           }
        self.datacollector = DataCollector(model_reporters=model_reporters,
                                           agent_reporters=agent_reporters)

        # Collect data for initialization phase
        self.datacollector.collect(self)

    logging.info("Model initialization complete.")

    # ------------------------------------------------ #
    # -------- MODEL INITIALIZATION FUNCTIONS -------- #
    # ------------------------------------------------ #
    
    def initialize_households(self, parcel_file):
        """Initialize households and assign properties. Create F_income --> price range table."""

        # Step 1: Load data and initialize parcels
        parcel_data = pd.read_csv(parcel_file)
        logging.info(f"Number of parcels loaded: {len(parcel_data)}")
        self.parcels = []
        for _, row in parcel_data.iterrows():
            self.add_parcel(row)

        # Step 2: Sort parcels and assign households by income groups
        n = len(self.parcels)
        sorted_parcels = sorted(self.parcels, key=lambda p: p.initial_price)
        self.households = []
        self.price_income_table = []
        self.fixed_income_group_thresholds = []

        for i, parcel in enumerate(sorted_parcels):
            group = (i * self.n_income_groups) // n + 1
            F_income = F_INCOME_BY_GROUP[group]
            price = parcel.initial_price
            income = np.round(price / (F_income * self.mortgage_years))

            hh = Household(
                model=self,
                income=income,
                F_income=F_income,
                income_group=group,
                price_margin_bounds=self.price_margin_bounds
            )
            hh.property = parcel
            parcel.owner = hh
            self.households.append(hh)
            self.schedule.add(hh)

        # Calculate the income distribution
        self.initialize_income_distribution()

    def add_parcel(self, parcel_chars):
        """Create a new Parcel and add it to the model.

        Notes
        -----
        `parcel_chars` can be a pandas Series (row from a dataframe) or a plain dict.
        We assign a stable `unique_id` using (priority order):
          1) an explicit id field in the row (common variants)
          2) the running index (= current length of self.parcels)
        """

        # Allow passing a pandas Series directly
        if hasattr(parcel_chars, "to_dict"):
            parcel_chars = parcel_chars.to_dict()

        # Best-effort ID extraction (keeps you compatible with different pipelines)
        uid = None
        for key in ("unique_id", "UNIQUE_ID", "parcel_id", "PARCEL_ID", "ID", "id"):
            if key in parcel_chars and parcel_chars[key] is not None:
                uid = int(parcel_chars[key])
                break

        if uid is None:
            uid = int(len(self.parcels))

        parcel = Parcel(uid, parcel_chars)
        self.parcels.append(parcel)
        return parcel
    
    def add_household(self, income, F_income, income_group, price_margin_bounds):
        """Create a new household agent with assigned income and income group.

        Args:
            income (float): Annual household income.
            F_income (float): Fraction of income allocated to housing (affordability ratio).
            income_group (int): Income group identifier (e.g., 1 to 5).

        Returns:
            Household: Initialized household agent.
        """
        hh = Household(
            model=self,
            income=income,
            F_income=F_income,
            income_group=income_group,
            price_margin_bounds=self.price_margin_bounds
        )

        self.schedule.add(hh)
        self.households.append(hh)
        return hh
    
    def initialize_income_distribution(self):
        """Initialize income groups, thresholds, and price-income mapping based on parcel prices."""

        if not self.households:
            logging.warning("No households to initialize income distribution.")
            return

        # Sort households by income
        self.households.sort(key=lambda hh: hh.income)
        incomes = np.array([hh.income for hh in self.households])
        income_bins = np.array_split(incomes, self.n_income_groups)

        # Save fixed thresholds and bounds
        self.fixed_income_group_thresholds = [group[-1] for group in income_bins[:-1]]
        self.fixed_income_group_bounds = [(group[0], group[-1]) for group in income_bins]

        # Create initial price-income mapping
        self.price_income_table = []
        for g in range(1, self.n_income_groups + 1):
            group_parcels = [p for i, p in enumerate(sorted(self.parcels, key=lambda p: p.initial_price))
                            if (i * self.n_income_groups) // len(self.parcels) + 1 == g]
            group_prices = [p.initial_price for p in group_parcels]
            group_incomes = [hh.income for hh in self.households
                            if (self.households.index(hh) * self.n_income_groups) // len(self.households) + 1 == g]

            if group_prices:
                self.price_income_table.append({
                    "Group": g,
                    "F_income": F_INCOME_BY_GROUP[g],
                    "Price_median": np.median(group_prices),
                    "Price_min": min(group_prices),
                    "Price_max": max(group_prices),
                    "Income_median": np.median(group_incomes),
                    "Income_min": min(group_incomes),
                    "Income_max": max(group_incomes),
                    "Count": len(group_parcels)
                })

        df = pd.DataFrame(self.price_income_table)
        logging.info(f"Fixed income thresholds and price-income mapping initialized:\n{df}")

    def dynamic_income_groups_distribution(self):
        """Update dynamic income group thresholds during the simulation."""

        if not self.households:
            logging.warning("No households available to update income distribution.")
            return

        self.households.sort(key=lambda hh: hh.income)
        incomes = np.array([hh.income for hh in self.households])
        income_bins = np.array_split(incomes, self.n_income_groups)

        self.dynamic_income_thresholds = [group[-1] for group in income_bins[:-1]]
        self.dynamic_income_bounds = [(group[0], group[-1]) for group in income_bins]

        logging.info(f"Dynamic income group bounds updated at step {self.schedule.steps}.")

    def record_income_group_counts(self, step, run):
        """Record number of households per income group by flood zone."""

        # Always use fixed thresholds for result analysis
        thresholds = self.fixed_income_group_thresholds

        # Initialize group counters
        flood_prone_counts = {f"FP_IncomeGroup_{i+1}": 0 for i in range(self.n_income_groups)}
        non_flood_prone_counts = {f"NonFP_IncomeGroup_{i+1}": 0 for i in range(self.n_income_groups)}

        for hh in self.households:
            income = hh.income
            flood_prone = hh.property.d_floodprone if hh.property else 0

            # Assign income group
            income_group = next((i + 1 for i, threshold in enumerate(thresholds) if income <= threshold), self.n_income_groups)

            if flood_prone:
                flood_prone_counts[f"FP_IncomeGroup_{income_group}"] += 1
            else:
                non_flood_prone_counts[f"NonFP_IncomeGroup_{income_group}"] += 1

        # Store result row
        row = {"Run": run, "Step": step, "Year": step / self.kY}
        row.update(flood_prone_counts)
        row.update(non_flood_prone_counts)
        self.income_group_data.append(row)


    def record_income_group_bounds(self, step, run):
        """Records current income group bounds."""
        if not hasattr(self, "dynamic_income_bounds") or not self.dynamic_income_bounds:
            return  # Skip if not calculated yet

        row = {"Run": run, "Step": step, "Year": step / self.kY}
        for i, (low, high) in enumerate(self.dynamic_income_bounds):
            row[f"Group_{i+1}_Low"] = low
            row[f"Group_{i+1}_High"] = high

        self.dynamic_income_bounds_history.append(row) 


    def export_income_statistics(self, group_counts_file, bounds_file=None):
        """Export recorded income group counts and (optional) dynamic bounds."""
        pd.DataFrame(self.income_group_data).to_csv(group_counts_file, index=False)
        logging.info(f"Income group counts exported to {group_counts_file}.")

        if bounds_file and hasattr(self, "dynamic_income_bounds_history"):
            pd.DataFrame(self.dynamic_income_bounds_history).to_csv(bounds_file, index=False)
            logging.info(f"Dynamic income bounds exported to {bounds_file}.")
        
    def remove_household(self, hh):
        """Remove a household object from the model. """
        self.households.remove(hh)
        self.schedule.remove(hh)
        del hh

    def update_parcel_age(self):
        """Increment the age of all parcels by 1 year."""
        for parcel in self.parcels:
            if "AGE" in parcel.prop_chars_raw:
                try:
                    parcel.prop_chars_raw["AGE"] = float(parcel.prop_chars_raw["AGE"]) + 1
                except Exception:
                    parcel.prop_chars_raw["AGE"] = 0.0

    
    # ------------------------------------------------ #
    # -------------- TRADING FUNCTIONS --------------- #
    # ------------------------------------------------ #
    def proportion_of_sellers(self):
        """
        Calculate the proportion of houses available at time t based on turnover rate.

        Returns:
        - proportion_of_sale (float): The proportion of houses available at time t.
        """
        turnover_rate = 1 / (self.avg_res_time_steps)
        mean_seller_frac = math.ceil((1 - math.exp(-turnover_rate)) * 100) / 100.0
        
        std = 0.005

        return (mean_seller_frac, std)
    
    def assign_sellers(self, F_seller):
        """
        Assigns households as sellers based on seller selection mode: a probabilistic function of residence time or random.
        Number of seller agents determined based on balance seller configuration:
            - Balance_seller = True, adaptive supply matching a desired steady-state seller share
            - Balance_seller = False, a fixed turnover probability (i.e. constant and continues supply not influenced by the existing number of sellers)

        Args:
            F_seller (tuple):     Mean and std dev of fraction of sellers.
            mode (str):         Seller selection mode ("Probabilistic" or "Random").
        """

        # Draw number of sellers
        self.seller_frac = np.round(self.rng.normal(F_seller[0], F_seller[1]),3)

        # List of eligible inactive owners
        owners = [hh for hh in self.households
                  if (hh.market_status == "Inactive" and
                  hh.moving_wait_time == 0)]
                
        # Determine number of new sellers
        if self.balance_sellers:
            # Maintain steady-state seller inventory
            current_sellers = sum(1 for hh in self.households if hh.market_status == "Seller")
            target_sellers = round(self.seller_frac * len(owners))
            n_sellers = max(0, target_sellers - current_sellers)
        else:
            # Assing new sellers regardless of existing listings
            n_sellers = max(0, round(self.seller_frac * len(owners)))

        if n_sellers == 0 or len(owners) == 0:
            return []

        # Select properties for sale based on seller mode
        if self.seller_mode == "Random":
            # Select random "Inactive" households to become sellers
            new_sellers = list(self.rng.choice(owners, n_sellers, replace=False))

        elif self.seller_mode == "Probabilistic":
                # Compute probability of selling based on residence time
                new_sellers = []
         

                for hh in owners:
                    # Compute how long the household has lived in the house
                    years_lived = self.schedule.steps - hh.purchase_step
                    residence_time = hh.res_time_steps  # Expected residence time

                    # Compute probability of selling (logistic function)
                    # Slope of the curve for function transitions from 0% probability to 100% probability
                    # (Gradual: -0.2, Moderate: -0.5, Sharp: -1.0)
                    k = -0.5
                    probability_to_sell = 1 / (1 + np.exp(k * (years_lived - residence_time)))
                    

                    # Draw a random number, sell if below probability threshold
                    if self.rng.uniform(0, 1) < probability_to_sell:
                        new_sellers.append(hh)

                # Ensure we select exactly `n_sellers`
                if len(new_sellers) > n_sellers:
                    new_sellers = list(self.rng.choice(new_sellers, n_sellers, replace=False))

        else:
            raise ValueError("Invalid mode. Use 'Random' or 'Probabilistic'.")

        # Change status of selected owners to sellers
        for hh in new_sellers:
            hh.market_status = "Seller"

        return new_sellers

    def create_new_buyers(self):
        """Create new buyer agents using resampling from the current market, assigning F_income based on fixed thresholds."""

        # Step 1: Calculate number of new buyers
        buyer_frac = self.seller_frac*self.buyer_demand
        n_parcels = len(self.parcels)
        n_buyers = round(self.new_buyer_coef * buyer_frac * n_parcels) if self.schedule.steps > 0 else round(buyer_frac * n_parcels)

        if n_buyers <= 0:
            logging.info("No new buyers entering this step.")
            return

        # Step 2: Prepare endogenous income samples from the current market
        current_incomes = np.array([hh.income for hh in self.households])

        if len(current_incomes) == 0:
            logging.warning("No existing household incomes to sample from!")
            return

        # Step 3: Sample new incomes uniformly from the current market distribution
        new_incomes = self.rng.choice(current_incomes, size=n_buyers, replace=True)

        new_buyers = []
        for income in new_incomes:
            # Step 4: Determine the income group using fixed thresholds
            group = next((i + 1 for i, threshold in enumerate(self.fixed_income_group_thresholds) if income <= threshold), self.n_income_groups)

            # Step 5: Assign F_income based on the initial (fixed) group mapping
            F_income = F_INCOME_BY_GROUP[group]

            hh = Household(
                model=self,
                income=income,
                F_income=F_income,
                income_group=group,
                price_margin_bounds=self.price_margin_bounds
            )
            hh.market_status = "Buyer"
            hh.budget = int(F_income * income * self.mortgage_years)
            hh.update_search_budget()

            self.schedule.add(hh)
            self.households.append(hh)
            new_buyers.append(hh)

        return new_buyers

    def register_transaction(self, prop, seller_id, buyer_id,
                         p_ask, p_bid, p_trans, highest_bid):
        """Register a trade transaction and update property price."""
        prop_id = getattr(prop, "unique_id", None)
        if prop_id is None:
            prop_id = id(prop)

        self.transactions[prop_id] = {
            "Property ID": prop_id,
            "Seller ID": seller_id,
            "Buyer ID": buyer_id,
            "P_ask": p_ask,
            "P_bid": p_bid,
            "P_trans": p_trans,
            "P_highest_bid": highest_bid,

            # Parcel attributes (use getattr to avoid crashes if missing)
            "P_floodprone": getattr(prop, "d_floodprone", None),
            "P_NbS_distance": getattr(prop, "DIST_MEUSE", None),
            "P_listing_time": getattr(prop, "listing_time", 0),

            # Computed components (may not exist if not computed for this parcel)
            "P_exp_flood_loss": getattr(prop, "expected_flood_loss", None),
            "P_amenity_value": getattr(prop, "amenity_value", None),
        }
    
        # Update last transaction price
        prop.record_sale(p_trans)


    def step(self, run_number):
        """Defines a single RHEA Model timestep. """

        # Step 1: Select new sellers from current inactive property owners
        self.assign_sellers(self.F_seller)

        # Step 2: Store properties for sale
        self.props_for_sale = [hh.property for hh in self.households if hh.market_status == "Seller"]

        # Step 3: Add new buyers
        if self.dynamic_income_groups:
            self.dynamic_income_groups_distribution()
        self.create_new_buyers()

        # Step 4: Execute trades (staged schedule)
        self.transactions = {}  # Clear transaction history
        self.schedule.step()
        logging.debug(f"Step {self.schedule.steps}: trades this step = {len(self.transactions)}")

        # Step 5: Update parcel age at end of year
        if self.schedule.steps % self.kY == 0:
            self.update_parcel_age()

        # Step 6: Update income distribution only if enabled
        if self.dynamic_income_groups:
            self.record_income_group_bounds(self.schedule.steps, run_number)

        # Step 7: Collect data
        self.record_income_group_counts(self.schedule.steps, run_number)
        self.datacollector.collect(self)

        print()  # Visual step separator for terminal

def log_model_parameters(model_kwargs):
    logging.info("=" * 40)
    logging.info("MODEL PARAMETERS")
    logging.info("=" * 40)
    for attr, value in model_kwargs.items():
        logging.info(f"{attr}: {value}")
    logging.info("=" * 40)

