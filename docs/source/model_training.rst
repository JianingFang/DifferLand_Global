Training
=====

Overview
--------

The training driver, ``experiments/calibration.py``, calibrates DifferLand end-to-end:

* Spatialization network → maps predictors to ecological parameters  
* DALEC TBM → simulates ecosystem dynamics  
* Loss function → compares simulations with observations  

Training is optimized with **Optax Adam** in JAX.

Required Input Files
--------------------

Available from the `Zenodo archive <https://zenodo.org/records/13984226?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk0YzRlMWU5LTlhZTYtNDJjZS1hOWVlLTVhNmNiOGY3YjFiMiIsImRhdGEiOnt9LCJyYW5kb20iOiJjNTEzNDNjNzQwNzIzZDQ1NGFiZTVmYzUzYTc3YzIyYyJ9.94jiR53dHwvL0HKZxXY7qNjKiIai9MlslzUGU_8Rugti8sRMfBdCoyykN7ooPLPrqew7sG7yH2ec1kv7s8LA>`_:

* ``differland_global_driver_v6.nc`` — spatial predictors, forcings, and targets  
* ``co2_mm_gl_01_23.csv`` — atmospheric CO₂ concentrations  
* ``era_valid_v6.nc`` — valid land mask  
* ``run_simulation_idx_v6.nc`` — shuffle/split indices  
* ``assimilate_bulk_variable_v6.nc`` — assimilation patches  

Set the paths in ``calibration.py``:

.. code-block:: python

   DATA_DIR = "/path/to/CARDAMOM_driver_data"
   DATA_DIR = os.path.join(DATA_DIR, "global/")

Command-Line Options
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-r, --run``
     - (required)
     - Run index (1–100). Controls shuffle/split seed.
   * - ``-p, --predictors``
     - (required)
     - Predictor set (e.g. ``PFT+CLIM+SOIL+AGE``).
   * - ``-n, --neurons``
     - 32
     - Hidden layer width of MLP blocks.
   * - ``-x, --hidden_layers``
     - 3
     - Number of hidden layers.
   * - ``-i, --iterations``
     - 199
     - Training iterations.
   * - ``-l, --learning_rate``
     - 5e-5
     - Adam learning rate.
   * - ``-t, --number_of_timesteps``
     - 23*12
     - Number of monthly timesteps (years × 12).
   * - ``-v, --verbose``
     - False
     - Print progress in addition to logging.

Outputs
-------

* Final parameters: ``./output/dalec993_<PREDICTORS>_run_<RUN>.pickle``  
* Checkpoints (every 10 iters): ``..._checkpoint-<iter>.pickle``  
* Logs: ``./log/dalec993_<PREDICTORS>_run_<RUN>.log``

How to Run
----------

1. **Prepare data paths** in ``calibration.py``.  
2. **Run training**:

.. code-block:: bash

   # Minimal example
   python calibration.py \
     --run 1 \
     --predictors PFT+CLIM+SOIL+AGE

1. **Monitor logs** in ``experiments/log/``. Validation loss is printed every 10 iterations.  

Adjusting Target Variables
--------------------------

The set of observational constraints used in training can be customized directly
in ``calibration.py`` by editing the target flags under:

.. code-block:: python

   ##### Define the target variables to be assimilated into the framework #####

   SIF_PROXY_TARGET = "LCSPP"       # LCSPP | none
   NBE_INVERSION_TARGET = "NBE"     # NBE (CMS-Flux) | CAMS_NBE | none
   SATELLITE_LAI_TARGET = "LAI"     # LAI (MODIS) | LAI_COPERNICUS | none
   GRACE_LWE_TARGET = "LWE_normalized"   # LWE_normalized (GRACE EWT Anomaly) | none
   BIOMASS_TARGET = "biomass_yan"   # biomass_yan (Xu et al. 2021) | biomass_ib (Li et al. 2025) | none
   FIRE_EMISSION_TARGET = "GFED_FIRE_EMISSION"  # GFED_FIRE_EMISSION | fire_emission (CMS-based inversion) | none
   STATIC_SOIL_TARGET = "som_const" # som_const (Harmonized World Soil Database) | none
   VOD_TARGET = "none"              # none (not assimilated)
   GPP_FLUXNET_TARGET = "gpp_fluxnet"   # gpp_fluxnet (PFT >25%) | gpp_fluxnet_10percent | gpp_fluxnet_50percent | none
   RECO_FLUXNET_TARGET = "reco_fluxnet" # reco_fluxnet (PFT >25%) | reco_fluxnet_10percent | reco_fluxnet_50percent | none
   ET_FLUXNET_TARGET = "et_fluxnet"     # et_fluxnet (PFT >25%) | et_fluxnet_10percent | et_fluxnet_50percent | none
   GLEAM_ET_TARGET = "ET"               # ET (GLEAM) | none

Each variable can be **enabled, switched to an alternative dataset, or disabled**
by changing its string value.

Examples
~~~~~~~~

* Assimilate only NBE (CMS-Flux) and MODIS LAI:

  .. code-block:: python

   SIF_PROXY_TARGET = "none"       
   NBE_INVERSION_TARGET = "NBE"     
   SATELLITE_LAI_TARGET = "LAI"     
   GRACE_LWE_TARGET = "none"   
   BIOMASS_TARGET = "none"   
   FIRE_EMISSION_TARGET = "none"  
   STATIC_SOIL_TARGET = "none" 
   GLEAM_ET_TARGET = "none"              
   VOD_TARGET = "none"              
   GPP_FLUXNET_TARGET = "none"   
   RECO_FLUXNET_TARGET = "none" 
   ET_FLUXNET_TARGET = "none"  

* Disable all eddy covariance constraints, but use remote sensing and top-down constraints:

  .. code-block:: python

   SIF_PROXY_TARGET = "LCSPP"       
   NBE_INVERSION_TARGET = "NBE"     
   SATELLITE_LAI_TARGET = "LAI"     
   GRACE_LWE_TARGET = "LWE_normalized"   
   BIOMASS_TARGET = "biomass_yan"   
   FIRE_EMISSION_TARGET = "GFED_FIRE_EMISSION"  
   STATIC_SOIL_TARGET = "som_const" 
   GLEAM_ET_TARGET = "ET"              
   VOD_TARGET = "none"              
   GPP_FLUXNET_TARGET = "none"   
   RECO_FLUXNET_TARGET = "none" 
   ET_FLUXNET_TARGET = "none"     

This flexibility allows DifferLand to be trained under different combinations
of observational streams depending on research needs.

Practical Tips
--------------

* Each ``--run`` index ensures reproducible splits. 
* Note that the neural networks however are independently initialized. You may adjust this behavior by setting the seeds in `init_mlp_params`.
* Checkpoints are written every 10 iterations but not auto-resumed.  
* Adjust ``--number_of_timesteps`` to match the driver dataset length.  

Example Checkpoints
--------------
* Example checkpoints are stored in `Zenodo archive <https://zenodo.org/records/13984226?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk0YzRlMWU5LTlhZTYtNDJjZS1hOWVlLTVhNmNiOGY3YjFiMiIsImRhdGEiOnt9LCJyYW5kb20iOiJjNTEzNDNjNzQwNzIzZDQ1NGFiZTVmYzUzYTc3YzIyYyJ9.94jiR53dHwvL0HKZxXY7qNjKiIai9MlslzUGU_8Rugti8sRMfBdCoyykN7ooPLPrqew7sG7yH2ec1kv7s8LA>`_ to reproduce the results in the manuscript.