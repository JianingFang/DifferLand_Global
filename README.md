# DifferLand (Global)
By Jianing Fang (jf3423@columbia.edu), last revised Jan 17, 2025.

This depository contains the source code for the DifferLand model described in the manuscript "*Differentiable Land Model Reveals Global Environmental Controls on Latent Ecological Functions*." This README file walks you through the process of installing and running the DifferLand package. 

## Repository Structure
- `DifferLand/`: the source code of the DifferLand model.
- `experiments/`: contains the scripts for training the model, saving the results, and computing test statistics and SHAP values.
- `data/`: contains the drive files for running the DifferLand model and analyzing the results. To reproduce the results in the manuscript, you can download the driver files from the [Zenodo repository](https://doi.org/10.5281/zenodo.13984225) and place them in this directory.

## Installation
1. Create a new conda environment and active the environment  
`conda create --name DifferLand python=3.12`  
`conda activate DifferLand`
2. Install the packages that DifferLand depends on:  
`conda install -c conda-forge jax numpy xarray pandas shap netCDF4 matplotlib cartopy scikit-learn`  
`pip install optax tqdm`  

3. Download the necessary driver files from the [Zenodo repository](https://doi.org/10.5281/zenodo.13984225) and place them in the `data\` directory.


## Model Calibration
4. First, go to the experiments directory. `cd experiments`
5. Run the model calibration.
For instance, if you want to calibrate the model with the full set of predictors ("PFT+CLIM+SOIL+AGE"), you can run the following command `python calibration.py -r 1 -p PFT+CLIM+SOIL+AGE` The training log will be stored in `experiments/log`, and the calibrated model parameters will be stored as a pickle file in `experiments/output` folder.  

## Compute Statistics and Analyze Results
6. Compute test statistics and output .nc files of parameter maps and model output of simulated carbon and water fluxes. For instance, `python evaluate_performance.py -r 1 -p PFT+CLIM+SOIL+AGE`.
7. Compute SHAP values by running the script compute_shap.py 

## Reference
Fang, J., Bowman, K., Zhao, W., Lian, X., Gentine, P. (2024). Differentiable Land Model Reveals Global Environmental Controls on Latent Ecological Functions. *Nature Communications*. In Press.

## Related projects:
In addition to the global model, we also have a [local DifferLand model](https://essopenarchive.org/users/747253/articles/1220863-exploring-optimal-complexity-for-water-stress-representation-in-terrestrial-carbon-models-a-hybrid-machine-learning-model-approach) suitable for site-level studies that can flexibly accommodate various machine learning based parameterizations. 




