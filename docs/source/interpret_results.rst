Evaluate Performance
=================

The script ``evaluate_performance.py`` is designed to evaluate DifferLand model
checkpoints after training. It loads saved neural network and DALEC parameters,
computes performance metrics against observational constraints, and exports both
latent parameters and simulated ecological variables for downstream analysis.

Main Tasks
----------

1. **Load training checkpoints**  
   - Reads parameter states from the ``output/`` directory (produced by ``calibration.py``).  
   - Supports both final parameter files and intermediate checkpoints.  

2. **Compute evaluation metrics**  
   - Compares simulated outputs against the observational datasets assimilated during training (e.g., NEE/NBE, GPP, RECO, LAI, ET, SIF, biomass, SOC, fire emissions).  
   - Reports standard error metrics (e.g., RMSE, correlation, bias) for validation and test sets.  The first 2 years are used for warm up and are typically excluded from the evaluation (the length of warm up period (in months) can be adjusted by setting the `--warm_up` option).

3. **Output results for analysis**  
   - Saves latent ecological parameters learned by the spatialization network.  
   - Saves simulated ecological variables (time series and grid-based) from DALEC.  
   - Outputs are written in standard formats for further statistical analysis or visualization (e.g., spatial mapping, temporal trend evaluation).  

Usage
-----

Typical usage requires specifying the run index and predictor set to match a
previous training run. For example:

.. code-block:: bash

   python evaluate_performance.py \
     --run 1 \
     --predictors PFT+CLIM+SOIL+AGE

This will load the corresponding checkpoint
(``./output/dalec993_PFT+CLIM+SOIL+AGE_run_1.pickle``), compute metrics, and
produce outputs in the ``./postanalysis/`` directory.

Outputs
-------

* **Metrics**: quantitative performance summaries for validation/test datasets.  
* **Latent ecological parameters**: retrieved parameter fields for mapping or
  sensitivity analysis.  
* **Ecological variables**: simulated time series and state/flux variables for
  subsequent analysis.  

These outputs provide the foundation for assessing DifferLand’s predictive skill
and for exploring ecological process representations across space and time.