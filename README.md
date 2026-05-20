# RHEA-NL: Agent-Based Housing Market Model under Flood Risk

## Overview
RHEA-NL is an agent-based housing market model developed to study how flood risk, housing market pressure, behavioral risk perception, and public climate adaptation strategies interact to shape urban housing market outcomes.

This repository contains the implementation of the model presented in [Mutlu and Filatova (2026)](https://www.sciencedirect.com/science/article/pii/S0198971526000426).

The model builds on the original RHEA framework [Filatova (2015)](https://www.sciencedirect.com/science/article/pii/S0198971514000714) and extends it with:
- Competitive bidding and endogenous price formation under varying market demand pressure
- Heterogeneous household flood risk perception and behavioral perception bias
- Public adaptation scenarios including traditional flood defenses and Nature-based Solutions (NbS)
- Spatial sorting and distributional dynamics in flood-prone housing markets

## Time Scale

- 1 timestep = 6 months
- Default simulation length = 30 years (60 steps)

## Key Findings

Using scenario experiments in a stylized Dutch housing market, the model shows that:

- Housing market scarcity and demand pressure are the primary drivers of exclusion and price growth
- Behavioral flood-risk underestimation weakens capitalization of objective flood risk into housing prices
- Public flood protection mainly reallocates housing demand spatially rather than substantially improving overall affordability
- Nature-based Solutions (NbS) produce modest additional amenity-driven price effects, while most exclusionary dynamics are driven by pre-existing housing scarcity and demand pressure

For detailed analysis, model calibration, sensitivity analysis, and discussion of limitations, see the associated publication.

## Repository Structure

| File | Description |
|---|---|
| `README.md` | Project overview, setup instructions, and model description |
| `run.py` | Main simulation runner |
| `model.py` | Core agent-based model logic |
| `household.py` | Household agent definitions and behavior |
| `parcel.py` | Parcel and housing unit representation |
| `realtor.py` | Realtor agent and pricing mechanisms |
| `single_run_scenarios.yaml` | Scenario configuration settings |
| `requirements.txt` | Python dependencies required to run the model |
| `dummy_dataset.csv` | Synthetic example dataset included for demonstration and testing purposes |

## Installation

git clone https://github.com/SC3-TUD/CEUS-RHEA_NL.git
cd CEUS-RHEA_NL
pip install -r requirements.txt

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Model

```bash
python run.py --config single_run_scenarios.yaml --scenario S1d
```
Available scenarios:
- S1a
- S1d
- S3

## Reproducibility

Due to data-sharing restrictions, the original NVM housing transaction data used in the paper cannot be redistributed. The repository therefore includes a synthetic dummy dataset to demonstrate the model workflow and code structure. Results reported in the paper require access to the calibrated empirical input dataset and scenario settings described in the publication.

### Expected data format

The input dataset should be provided as a CSV file, where each row represents a housing unit.

Example structure:

| AGE | HOUSESIZE | LOTSIZE | ROOMS | QUALITY | LN_DIST_CBD | LN_DIST_MEUSE | DIST_MEUSE | FP_PROTECTED | INIT_PRICE_2020 |
|-----|-----------|---------|-------|---------|-------------|---------------|------------|--------------|-----------------|
| ... | ...       | ...     | ...   | ...     | ...         | ...           | ...        | 0/1          | ...             |


## License

MIT License

## Citation

If you use this code, please cite:

> Mutlu, A., & Filatova, T. (2026). *Urban housing markets under flood risk: Modeling demand pressure, risk perception bias, and public interventions*. Computers, Environment and Urban Systems, 128, 102440. https://doi.org/10.1016/j.compenvurbsys.2026.102440

## Contact

**Asli Mutlu**  
PhD Researcher, Delft University of Technology (TU Delft)

- GitHub: [@asli-mutlu](https://github.com/asli-mutlu)
- LinkedIn: [linkedin.com/in/asli-mutlu](https://www.linkedin.com/in/asli-mutlu/)
- Email: a.mutlu@tudelft.nl
