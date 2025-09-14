Getting Started
====================

Download DifferLand
-------------------

DifferLand package is available on `GitHub <https://github.com/JianingFang/DifferLand_Global.git>`_.  
You can clone the repository by running:

.. code-block:: bash

   git clone https://github.com/JianingFang/DifferLand_Global.git


Repository Structure
--------------------

- ``DifferLand/``: the source code of the DifferLand model.
- ``experiments/``: contains the scripts for training the model, saving the results, 
  and computing test statistics and SHAP values.
- ``data/``: contains the driver files for running the DifferLand model and analyzing the results.  
  To reproduce the results in the manuscript, you can download the driver files from the 
  `Zenodo repository <https://zenodo.org/records/13984226?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk0YzRlMWU5LTlhZTYtNDJjZS1hOWVlLTVhNmNiOGY3YjFiMiIsImRhdGEiOnt9LCJyYW5kb20iOiJjNTEzNDNjNzQwNzIzZDQ1NGFiZTVmYzUzYTc3YzIyYyJ9.94jiR53dHwvL0HKZxXY7qNjKiIai9MlslzUGU_8Rugti8sRMfBdCoyykN7ooPLPrqew7sG7yH2ec1kv7s8LAxQ>`_
  and place them in this directory.
- ``notebooks/``: contains Jupyter notebooks for analyzing results and plotting figures.


Experiment Setup
----------------

1. Create a new conda environment and activate it:

   .. code-block:: bash

      conda create --name DifferLand python=3.12
      conda activate DifferLand

2. Install the packages that DifferLand depends on:

   .. code-block:: bash

      conda install -c conda-forge jax numpy xarray pandas shap netCDF4 matplotlib cartopy scikit-learn
      pip install optax tqdm d

3. Download the necessary driver files from the 
   `Zenodo repository <https://doi.org/10.5281/zenodo.13984225>`_
   and place them in the ``data/`` directory.