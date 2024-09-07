# Membrane Process Modeling - Optimization
(meprom-opti)

Created by: Aron K. Beke

This repository contains code to support the manuscript "Multiobjective optimization and process intensification of multistage nanofiltration with structural synergy for brine separation" by A. K. Beke et al. (DOI:XX/XYZ).

The repository performs the multi-objective optimization of a four-stage nanofiltration system for the separation of monovalent and bivalent ions according to the details provided in the manuscript. Running run_optimization.py from terminal generates the Pareto-optimal operating points of the separation system according to the input parameters provided in a specified input text file, and returns a csv file with the permeate pressure, interstage dilution, and permeate recycling split ratios corresponding to the optimal operating points.

# Installation
Performing analysis using this package requires the installation of conda and dependencies.
1. Install conda from https://conda.io/projects/conda/en/latest/user-guide/install/index.html.
2. Download and save this package.
3. Navigate to the repository. Create an environment and install necessary dependencies easily from the environment.yml file, using the following terminal command: `conda env create -f environment.yml`. The environment will be named `meprom-opti`.
4. Activate the environment with `conda activate meprom-opti`.
5. Run analysis using `python run_optimization.py`.

# run_optimization.py
Running run_optimization.py will prompt the user for the name or path of a .txt file containing all necessary separation parameters. The input_parameters.txt file is provided as template. The code performs the following:
1. Optimization for 10 equidistant Pareto constraint levels between the specified minimal and maximal constraint levels.
2. Validation of the found solutions through simulation.
3. Selection of Pareto-optimal solutions.
After performing the optimizations, the code will output 3 csv files in the results folder:
1. One file with all the found solutions (might not be locally optimal)
2. One file with all the found solutions and their validations.
3. One file with the Pareto-optimal solutions.

Please provide the necessary input parameters according to the following considerations:
- Objective: "separation_factor" or "molar_power". If separation factor is chosen as objective, minimal divalent ion recovery will be applied as Pareto constraint. If molar power is chosen as objective, minimal separation factor will be applied as Pareto constraint.
- Maximum pressure has to be given in bar units.
- Number of models: the number of random initiations for multistart optimization 
- Number of nodes: the number of stirred control volumes-in-series to be applied in modeling the hydrodynamics of a membrane separation module.
- Min. constraint: the minimum Pareto constraint level (recovery or separation factor) to be examined.
- Max. constraint: the maximum Pareto constraint level (recovery or separation factor) to be examined.
- Pressure exchange, interstage dilution, permeate recycling: should be set to 0 or 1, depending on whether the intensification approach should be included or not in the model.