# DifferLand (Global)
By Jianing Fang (jf3423@columbia.edu), last revised Oct 22, 2024.

This depository contains the source code for the DifferLand model described in the manuscript "*Differentiable Land Model Reveals Global Environmental Controls on Latent Ecological Functions*." This README file walks you through the process of installing and running the DifferLand package. 

## Repository Structure
- `DifferLand/`: the source code of the DifferLand model.
- `experiments/`: contains the scripts for training the model, saving the results, and computing test statistics and SHAP values.
- `data/`: contains the drive files for running the DifferLand model and analyzing the results. To reproduce the results in the manuscript, you can download the driver files from the [Zenodo repository](https://zenodo.org/records/13984226?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk0YzRlMWU5LTlhZTYtNDJjZS1hOWVlLTVhNmNiOGY3YjFiMiIsImRhdGEiOnt9LCJyYW5kb20iOiJjNTEzNDNjNzQwNzIzZDQ1NGFiZTVmYzUzYTc3YzIyYyJ9.94jiR53dHwvL0HKZxXY7qNjKiIai9MlslzUGU_8Rugti8sRMfBdCoyykN7ooPLPrqew7sG7yH2ec1kv7s8LAxQ) and place them in this directory.
- `notebooks/`: contains jupyter notebooks for analyzing results and plotting figures.

## Installation
1. Create a new conda environment and active the environment  
`conda create --name DifferLand python=3.9`  
`conda activate DifferLand`
2. Install the packages that DifferLand depends on:  
`conda install -c conda-forge jax numpy xarray pandas shap netCDF4 matplotlib cartopy scikit-learn`  
`pip install fastkde optax tqdm`  
(Optional) if you also want to run the code in the jupyter notebooks, you will need to make sure that jupyter is installed.
3. Download the necessary driver files from the [Zenodo repository](https://zenodo.org/records/13984226?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk0YzRlMWU5LTlhZTYtNDJjZS1hOWVlLTVhNmNiOGY3YjFiMiIsImRhdGEiOnt9LCJyYW5kb20iOiJjNTEzNDNjNzQwNzIzZDQ1NGFiZTVmYzUzYTc3YzIyYyJ9.94jiR53dHwvL0HKZxXY7qNjKiIai9MlslzUGU_8Rugti8sRMfBdCoyykN7ooPLPrqew7sG7yH2ec1kv7s8LAxQ) and place them in the `data\` directory.


## Model Calibration
4. First, go to the experiments directory. `cd experiments`
5. Run the model calibration.
For instance, if you want to calibrate the model with the full set of predictors ("PFT+CLIM+SOIL+AGE"), you can run the following command `python calibration.py -r 1 -p PFT+CLIM+SOIL+AGE` The training log will be stored in `experiments/log`, and the calibrated model parameters will be stored as a pickle file in `experiments/output` folder.  
This is what the training print out will look like:
```
Welcome to DifferLand!
Folder path exists: ./output/
Folder path exists: ./log/
Number of hidden layers in the embeeding NN: 3
Number of neurons in each NN layer: 32
Total number of training iterations: 199
Learning rate: 5e-05
Number of timesteps: 168
Spatial predictors:
+ LAT
+ NF
+ DBF
+ EBF
+ MF
+ SH
+ SAV
+ GRA
+ WET
+ CRO
+ NVG
+ MAT
+ MAP
+ elevation
+ canopy_height
+ tree_age_000_filled
+ BULK_DEN
+ SAND
+ SILT
+ CLAY
+ GRAVEL
Reading in datasets for model training...
Data for model calibration have been successfully loaded.
Initializing model parameters...
Training start!
100%|███████████████████████████████████████████████████████████████████████| 199/199 [5:18:53<00:00, 96.15s/it]
Training complete!
Calibrated parameters saved to: ./output/dalec993_PFT+CLIM+SOIL+AGE_run_1.pickle
```
Training this model for 199 iterations took about 5.5 hours on my personal laptop (i.e., MacBook Pro with 2 GHz Quad-Core Intel Core i5 CPU and 16GB Memory). It can be parallized on a HPC cluster for ensemble training.
## Compute Statistics and Analyze Results
6. Compute test statistics and output .nc files of parameter maps and model output of simulated carbon and water fluxes. For instance, `python compute_r2.py -r 1 -p PFT+CLIM+SOIL+AGE`.
7. Compute SHAP values by running the script compute_shap.py (e.g. python compute_shap.py -r 1 -p PFT+CLIM+SOIL+AGE).
8. Analyze the results using the jupyter notebooks provided in the `notebooks/` directory, or customize the code for your own applications. Note that you will need to download some additional datasets to run all the code in the notebooks. Details about which datasets we used are described in the manuscript. You may also need to adjust some path names for your own environment. 

## Reference
Fang, J., Bowman, K., Zhao, W., Lian, X., Gentine, P. (2024). Differentiable Land Model Reveals Global Environmental Controls on Latent Ecological Functions. *Submitted*.

## Related projects:
In addition to the global model, we also have a [local DifferLand model](https://essopenarchive.org/users/747253/articles/1220863-exploring-optimal-complexity-for-water-stress-representation-in-terrestrial-carbon-models-a-hybrid-machine-learning-model-approach) suitable for site-level studies that can flexibly accommodate various machine learning based parameterizations. 




