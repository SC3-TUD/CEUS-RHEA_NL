# RHEA-NL: Agent-Based Housing Market Model under Flood Risk

Author: Asli Mutlu

## Overview
This repository contains the code for the RHEA-NL agent-based model, developed to study housing market dynamics under flood risk and climate adaptation strategies.

The model builds on the original RHEA framework and introduces:
- A rolling hedonic price mechanism via a Realtor agent
- Behavioral flood risk perception
- Policy scenarios including Nature-based Solutions (NbS)

The model is designed to analyze distributional effects, price dynamics, and spatial distribution of households and housing demand under different adaptation strategies.

Time scale: - 1 timestep = 6 months - Default simulation length = 30
years (60 steps)

## Repository Structure

├── run.py
├── model.py
├── household.py
├── parcel.py
├── realtor.py
├── single_run_scenarios.yaml
├── requirements.rhea-min.txt
├── environment.rhea-min.pinned.yml
└── data/ (not included)

## Installation

Clone the repository:
git clone <your-repository-link>
cd RHEA-NL

Install dependencies:
pip install -r requirements.txt

## Running the Model

python run.py --config single_run_scenarios.yaml --scenario S1d

Available scenarios:
- S1a
- S1d
- S3

## Data Availability

The original housing transaction data are provided by NVM and are subject to a non-disclosure agreement. Therefore, these data cannot be shared publicly.

This repository includes a synthetic/dummy dataset for demonstration purposes that does not represent real transactions.

Users must provide their own dataset for meaningful outcomes. 

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

Mutlu, A., Filatova, T. (2026). Urban Housing Markets under Flood Risk...

## Contact

Asli Mutlu (a.mutlu@tudelft.nl)
Delft University of Technology
