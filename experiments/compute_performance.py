import argparse

parser = argparse.ArgumentParser(
                    prog='Evaluate DifferLand Performance.',
                    description='Post-processing DifferLand runs by computing model performance metrics and output select modeled variables.',
                    epilog='By Jianing Fang')

parser.add_argument('-r', '--run', help="Enter an index for the run, used to identify independent calibrations (1-100)", type=int, required=True)
parser.add_argument('-p', '--predictors', required=True)
parser.add_argument('-n', '--neurons', default=32)
parser.add_argument('-x', '--hidden_layers', default=3)
parser.add_argument('-i', '--iterations', default=199, type=int)
parser.add_argument('-l', '--learning_rate', default=5e-5)
parser.add_argument('-t', '--number_of_timesteps', default=168, type=int)
parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, default=True)

output_dir = "./output/"
log_dir = "./log/"
postanalysis_dir = "./postanalysis/"
fig_dir = os.path.join(postanalysis_dir, "figure/")
nc_dir = os.path.join(postanalysis_dir, "nc/")
npy_dir = os.path.join(postanalysis_dir, "npy/")
metrics_dir = os.path.join(postanalysis_dir, "metrics/")
data_dir = "../data/"
args = parser.parse_args()

# parse command line arguments
if args.verbose:
    print("Now start post-processing DifferLand output...")

def get_predictor_list(predictor_set):
    predictor_list = ["LAT"]
    if "PFT" in predictor_set:
        predictor_list += ["NF", "DBF", "EBF", "MF", "SH", "SAV", "GRA", "WET", "CRO", "NVG"]
    if "CLIM" in predictor_set:
        predictor_list += ["MAT", "MAP", "elevation"]
    if "AGE" in predictor_set:
        predictor_list += ["canopy_height", "tree_age_000_filled"]
    if "SOIL" in predictor_set:
        predictor_list += ["BULK_DEN", "SAND", "SILT", "CLAY", "GRAVEL"]
    if "LATLON" in predictor_set:
        predictor_list += ["lat_deg", "lon_deg"]
    if "CONTROL" in predictor_set:
        predictor_list += ["null"]
        if len(predictor_list) == 1:
            print("Error: invalid predictor list. The predictor list must contain one or more from the set {PFT, CLIM, SOIL, AGE}, or it must be either LATLON or CONTROL.")
            exit()
    return predictor_list

HIDDEN_LAYERS = args.hidden_layers
NEURONS = args.neurons
TOTAL_ITER = args.iterations + 1
LEARNING_RATE = args.learning_rate
NT = args.number_of_timesteps
predictor_list = get_predictor_list(args.predictors)
if args.run < 1 or args.run > 100:
    print("Error: RUN_IDX must be an integer between 1 and 100 inclusive.")
    exit()
run = args.run

exp_str = "dalec993_{}_run_{}".format(args.predictors,
                                run)


if args.verbose:
    print("Number of hidden layers in the embeeding NN: {}".format(HIDDEN_LAYERS))
    print("Number of neurons in each NN layer: {}".format(NEURONS))
    print("Total number of training iterations: {}".format(TOTAL_ITER-1))
    print("Learning rate: {}".format(LEARNING_RATE))
    print("Number of timesteps: {}".format(NT))
    print("Spatial predictors:")
    for p in predictor_list:
        print("+ {}".format(p))

import jax
import jax.numpy as jnp
from functools import partial
import os
import numpy as np
import pandas as pd
import pickle
import fastkde
import sys
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xarray as xr
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy
import cartopy
import cartopy.crs as ccrs
from copy import deepcopy


sys.path.insert(1, '..')
from DifferLand.util.preprocessing import read_variable_to_vector, read_multiple_varible_to_array
from DifferLand.util.preprocessing import nan_read_multiple_variable_temporal_to_vector
from DifferLand.util.preprocessing import read_multiple_variable_temporal_to_vector
from DifferLand.util.preprocessing import generate_data_loader, generate_input_loader
from DifferLand.optimization.forward import embed_prediction_forward, parameter_prediction_forward
from DifferLand.optimization.loss_functions import *
from DifferLand.model.DALEC993 import DALEC993
from DifferLand.util.preprocessing import create_folder_if_not_exists
create_folder_if_not_exists(postanalysis_dir, verbose=args.verbose)
create_folder_if_not_exists(fig_dir, verbose=args.verbose)
create_folder_if_not_exists(nc_dir, verbose=args.verbose)
create_folder_if_not_exists(npy_dir, verbose=args.verbose)
create_folder_if_not_exists(metrics_dir, verbose=args.verbose)



rcParams['font.family'] = 'Inter'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = [5.0, 5.0]
rcParams['figure.dpi'] = 200
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'    


def forward(params, predictors, met, model):
    embedded_param_state, dalec_param_state, initial_param_state, pheno_param_state = params
    embedded_layer = embed_prediction_forward(embedded_param_state, predictors[1:])
    dalec_parameters = model.unnormalize(parameter_prediction_forward(dalec_param_state, embedded_layer))
    augmented_pheno_predictors = jnp.concatenate([embedded_layer, predictors[0:1]])
    pheno_parameters = model.unnormalize_pheno(parameter_prediction_forward(pheno_param_state, augmented_pheno_predictors))
    initial_pools = model.unnormalize_pools(parameter_prediction_forward(initial_param_state, embedded_layer))
    final_state, all_fluxes = jax.lax.scan(partial(model.step, 
                                                   gpp_params=None, 
                                                   pheno_parameters=pheno_parameters,
                                                   dalec_parameters=dalec_parameters), initial_pools, met)
    return all_fluxes


model = DALEC993(water_stress_type="default")

forward = partial(forward, model=model)
batch_forward = jax.jit(jax.vmap(jax.jit(forward), in_axes=[None, 0, 0]))


def compute_forward_run(model_name, predictor_matrix, met_patrix):
    with open(model_name, "rb") as fp:
        param_state = pickle.load(fp)
    predicted = batch_forward(param_state, predictor_matrix, met_patrix).squeeze()
    return predicted
    
def compute_patch_level_nee(model_name, predictor_loader, met_loader, label_loader, model, warm_up=0):
    with open(model_name, "rb") as fp:
        param_state = pickle.load(fp)
        
    nee_predicted = []
    nee_label = []
    sif_label = []

    for predictor, met, label in zip(predictor_loader, met_loader, label_loader):
        predicted_sample = batch_forward(param_state, predictor, met).squeeze()
        nee_predicted.append(np.sum(predicted_sample[:, warm_up:, model.pfn.nee] * label[:, warm_up:, 3], axis=0) / np.sum(label[:, warm_up:, 3], axis=0))
        nee_label.append(np.sum(label[:, warm_up:, 2] * label[:, warm_up:, 3], axis=0) / np.sum(label[:, warm_up:, 3], axis=0))
        sif_label.append(np.sum(label[:, warm_up:, 0] * label[:, warm_up:, 1], axis=0) / np.sum(label[:, warm_up:, 1], axis=0))
    
    nee_predicted_patch = np.stack(nee_predicted)
    nee_label_patch = np.stack(nee_label)
    sif_label_patch = np.stack(sif_label)
    return nee_predicted_patch, nee_label_patch, sif_label_patch


def compute_patch_level_water_pool(model_name, predictor_loader, met_loader, label_loader, model, warm_up=0):
    with open(model_name, "rb") as fp:
        param_state = pickle.load(fp)
        
    water_predicted = []
    water_label = []
    sif_label = []

    for predictor, met, label in zip(predictor_loader, met_loader, label_loader):
        predicted_sample = batch_forward(param_state, predictor, met).squeeze()

        water_series = predicted_sample[:, warm_up, model.pfn.next_water_pool]

        predicted_water_pool_mean = jnp.sum(water_series * label[:, warm_up:, 7], axis=1) / (np.sum(label[:, warm_up:, 7], axis=1) + 0.1)
        predicted_water_pool_normalized = (water_series.T-predicted_water_pool_mean).T
        predicted_water_pool_normalized = np.mean(predicted_water_pool_normalized * label[:, warm_up:, 7], axis=0)
        predicted_water_pool_normalized = jnp.where(jnp.mean(label[:, warm_up:, 7], axis=0) < 0.8, jnp.nan, predicted_water_pool_normalized)
        label_water_normalized = np.mean(label[:, warm_up:, 6] * label[:, warm_up:, 7], axis=0)
        label_water_normalized = jnp.where(jnp.mean(label[:, warm_up:, 7], axis=0) < 0.7, jnp.nan, label_water_normalized)
        
        label_sif_patch = jnp.sum(label[:, warm_up:, 0] * label[:, warm_up:, 1], axis=0) / jnp.sum(label[:, warm_up:, 1], axis=0)
        
        water_predicted.append(predicted_water_pool_normalized)
        water_label.append(label_water_normalized)
        sif_label.append(label_sif_patch)
    
    water_predicted_patch = np.stack(water_predicted)
    water_label_patch = np.stack(water_label)
    sif_label_patch = np.stack(sif_label)
    return water_predicted_patch, water_label_patch, sif_label_patch

def nan_filtered_r2_score(targets, predictions):
    sel = np.invert(np.isnan(targets) | np.isnan(predictions))
    if np.sum(sel) > 0:
        return r2_score(targets[sel], predictions[sel])
    else:
        return np.nan
def nse_score(targets, predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))
            

if args.verbose:
    print("Loading datasets...")

RUN_SIMULATION_IDX = read_variable_to_vector(data_dir, "run_simulation_idx_v5.4.nc", "run_simulation_idx", time_idx=run-1)
    
VALID = read_variable_to_vector(data_dir, "era_valid_v5.4.nc", "era_valid")
INVALID = np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID) | (RUN_SIMULATION_IDX < 0) # filter out dev PIXELS
TEST = np.invert(np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID)) & (RUN_SIMULATION_IDX < 0)
    
predictor_matrix = read_multiple_varible_to_array(data_dir, "differland_global_driver_v5.4.nc", predictor_list)
test_predictor_matrix = deepcopy(predictor_matrix)
test_predictor_matrix[:, np.invert(TEST)] = np.nan

predictor_matrix[:, INVALID] = np.nan
ASSIMILATE_BULK_FLAG = read_variable_to_vector(data_dir, "assimilate_bulk_variable_v5.4.nc", "assimilate_bulk_variable")

# filter out nan pixles
not_nan_idx = np.invert((np.sum(np.isnan(predictor_matrix), axis=0) > 0))
test_not_nan_idx = np.invert((np.sum(np.isnan(test_predictor_matrix), axis=0) > 0))


predictor_matrix = predictor_matrix[:, not_nan_idx]
predictor_matrix_test = test_predictor_matrix[:, test_not_nan_idx]

VALID_IDX = RUN_SIMULATION_IDX[not_nan_idx]
TEST_VALID_IDX = RUN_SIMULATION_IDX[test_not_nan_idx]

shuffle_idx = np.argsort(VALID_IDX, kind='mergesort')
test_shuffle_idx = np.argsort(TEST_VALID_IDX, kind='mergesort')


ASSIMILATE_SHUFFLE_FLAG = ASSIMILATE_BULK_FLAG[not_nan_idx][shuffle_idx]
TEST_ASSIMILATE_SHUFFLE_FLAG = ASSIMILATE_BULK_FLAG[test_not_nan_idx][test_shuffle_idx]

sorted_valid_idx = VALID_IDX[shuffle_idx]
predictor_matrix_shuffled = predictor_matrix[:, shuffle_idx]
sorted_valid_idx_test = TEST_VALID_IDX[test_shuffle_idx]

train_dev_idx = np.round(np.max(sorted_valid_idx) * 0.9).astype(np.int32)

sorted_valid_idx_train = sorted_valid_idx[sorted_valid_idx <= train_dev_idx]
sorted_valid_idx_dev = sorted_valid_idx[sorted_valid_idx > train_dev_idx]


predictor_matrix_train = predictor_matrix_shuffled[:, sorted_valid_idx <= train_dev_idx]
predictor_matrix_dev = predictor_matrix_shuffled[:, sorted_valid_idx > train_dev_idx]
predictor_matrix_test = predictor_matrix_test[:, test_shuffle_idx]

scaled_predictor_matrix_train = (predictor_matrix_train.T - np.mean(predictor_matrix_train, axis=1)) / np.std(predictor_matrix_train, axis=1)
scaled_predictor_matrix_dev = (predictor_matrix_dev.T - np.mean(predictor_matrix_train, axis=1)) / np.std(predictor_matrix_train, axis=1)
scaled_predictor_matrix_test = (predictor_matrix_test.T - np.mean(predictor_matrix_train, axis=1)) / np.std(predictor_matrix_train, axis=1)

train_matrix = jnp.array(scaled_predictor_matrix_train, dtype=jnp.float32)
dev_matrix = jnp.array(scaled_predictor_matrix_dev, dtype=jnp.float32)
test_matrix = jnp.array(scaled_predictor_matrix_test, dtype=jnp.float32)

met_list = ["DAYS", "T_min", "T_max", "SOLR", "CO2", "DOY", "BURNED_AREA", "VPD", "PREC", "LAT", "DELTA_T", "MAT", "MAP"]
met_matrix = read_multiple_variable_temporal_to_vector(data_dir, "differland_global_driver_v5.4.nc", met_list, not_nan_idx, shuffle_idx, n_t=NT)
met_matrix = jnp.transpose(met_matrix, axes=[2, 1, 0])
met_matrix_train = jnp.array(met_matrix[sorted_valid_idx <= train_dev_idx, :], dtype=jnp.float32)
met_matrix_dev = jnp.array(met_matrix[sorted_valid_idx > train_dev_idx, :], dtype=jnp.float32)

test_met_matrix = read_multiple_variable_temporal_to_vector(data_dir, "differland_global_driver_v5.4.nc", met_list, test_not_nan_idx, test_shuffle_idx, n_t=NT)
met_matrix_test = jnp.array(jnp.transpose(test_met_matrix, axes=[2, 1, 0]), dtype=jnp.float32)

output_list = ["SIF", "NBE", "LAI", "LWE_normalized", "agb_yan", "fire_emission", "som_const", "VOD", "gpp_fluxnet", "reco_fluxnet", "et_fluxnet", "ET"]

    
output_matrix = nan_read_multiple_variable_temporal_to_vector(data_dir, "differland_global_driver_v5.4.nc", output_list, not_nan_idx, shuffle_idx, n_t=NT)
output_matrix = jnp.transpose(output_matrix, axes=[2, 1, 0])

output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 2].set(-9999)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 3].set(0)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 6].set(-9999)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 7].set(0)
output_matrix_train = jnp.array(output_matrix[sorted_valid_idx <= train_dev_idx, :], dtype=jnp.float32)
output_matrix_dev = jnp.array(output_matrix[sorted_valid_idx > train_dev_idx, :], dtype=jnp.float32)

output_matrix_test = nan_read_multiple_variable_temporal_to_vector(data_dir, "differland_global_driver_v5.4.nc", output_list, test_not_nan_idx, test_shuffle_idx, n_t=NT)
output_matrix_test = jnp.transpose(output_matrix_test, axes=[2, 1, 0])
output_matrix_test=output_matrix_test.at[np.invert(TEST_ASSIMILATE_SHUFFLE_FLAG), :, 2].set(-9999)
output_matrix_test=output_matrix_test.at[np.invert(TEST_ASSIMILATE_SHUFFLE_FLAG), :, 3].set(0)
output_matrix_test=output_matrix_test.at[np.invert(TEST_ASSIMILATE_SHUFFLE_FLAG), :, 6].set(-9999)
output_matrix_test=output_matrix_test.at[np.invert(TEST_ASSIMILATE_SHUFFLE_FLAG), :, 7].set(0)
output_matrix_test = jnp.array(output_matrix_test, dtype=jnp.float32)


train_MET = generate_data_loader(met_matrix_train, sorted_valid_idx_train, zero_padding=False)
dev_MET = generate_data_loader(met_matrix_dev, sorted_valid_idx_dev, zero_padding=False)
test_MET = generate_data_loader(met_matrix_test, sorted_valid_idx_test, zero_padding=False)


train_Y = generate_data_loader(output_matrix_train, sorted_valid_idx_train, zero_padding=True)
dev_Y = generate_data_loader(output_matrix_dev, sorted_valid_idx_dev, zero_padding=True)
test_Y = generate_data_loader(output_matrix_test, sorted_valid_idx_test, zero_padding=True)


train_X = generate_input_loader(train_matrix, sorted_valid_idx_train)
dev_X = generate_input_loader(dev_matrix, sorted_valid_idx_dev)
test_X = generate_input_loader(test_matrix, sorted_valid_idx_test)

if args.verbose:
    print("Data loaded")


pickle_name = os.path.join(output_dir, exp_str + ".pickle")

if args.verbose:
    print("loading calibrated parameters from {}".format(pickle_name))
with open(pickle_name, "rb") as fp:
    param_state = pickle.load(fp)

if args.verbose:
    print("Parameters loaded.")
    print("Now calculating model performance...")
    
predicted_matrix_train = compute_forward_run(pickle_name, train_matrix, met_matrix_train)
predicted_matrix_dev = compute_forward_run(pickle_name, dev_matrix, met_matrix_dev)
predicted_matrix_test = compute_forward_run(pickle_name, test_matrix, met_matrix_test)


nee_predicted_train = []
nee_label_train = []

water_predicted_train = []
water_label_train = []

fire_predicted_train = []
fire_label_train = []


warm_up=0

for k in range(len(train_X)):
    predicted_sample = batch_forward(param_state, train_X[k], train_MET[k]).squeeze()
    nee_predicted_train.append(np.sum(predicted_sample[:, warm_up:, model.pfn.nee] * train_Y[k][:, warm_up:, 3], axis=0) / np.sum(train_Y[k][:, warm_up:, 3], axis=0))
    nee_label_train.append(np.sum(train_Y[k][:, warm_up:, 2] * train_Y[k][:, warm_up:, 3], axis=0) / np.sum(train_Y[k][:, warm_up:, 3], axis=0))
    predicted_water_pool_mean = jnp.sum((predicted_sample[:, warm_up:, model.pfn.next_water_pool] ) * train_Y[k][:, warm_up:, 7], axis=1) / (np.sum(train_Y[k][:, warm_up:, 7], axis=1) + 0.1)
    predicted_water_pool_normalized = ((predicted_sample[:, warm_up:, model.pfn.next_water_pool] ).T-predicted_water_pool_mean).T
    predicted_water_pool_normalized = np.mean(predicted_water_pool_normalized * train_Y[k][:, warm_up:, 7], axis=0)
    predicted_water_pool_normalized = jnp.where(jnp.mean(train_Y[k][:, warm_up:, 7], axis=0) < 0.7, jnp.nan, predicted_water_pool_normalized)
    label_water = np.mean(train_Y[k][:, warm_up:, 6] * train_Y[k][:, warm_up:, 7], axis=0)
    label_water = jnp.where(jnp.mean(train_Y[k][:, warm_up:, 7], axis=0) < 0.7, jnp.nan, label_water)
    water_predicted_train.append(predicted_water_pool_normalized)
    water_label_train.append(label_water)
    fire_predicted_train.append(np.sum(predicted_sample[:, warm_up:, model.pfn.total_fire_combust] * train_Y[k][:, warm_up:, 11], axis=0) / np.sum(train_Y[k][:, warm_up:,11], axis=0))
    fire_label_train.append(np.sum(train_Y[k][:, warm_up:, 10] * train_Y[k][:, warm_up:, 11], axis=0) / np.sum(train_Y[k][:, warm_up:, 11], axis=0))
    
nee_predicted_train = np.stack(nee_predicted_train)
nee_label_train = np.stack(nee_label_train)

fire_predicted_train = np.stack(fire_predicted_train)
fire_label_train = np.stack(fire_label_train)

water_predicted_train = np.stack(water_predicted_train)
water_label_train = np.stack(water_label_train)

nee_predicted_test = []
nee_label_test = []

water_predicted_test = []
water_label_test = []

fire_predicted_test = []
fire_label_test = []

for k in range(len(test_X)):
    predicted_sample = batch_forward(param_state, test_X[k], test_MET[k]).squeeze()
    nee_predicted_test.append(np.sum(predicted_sample[:, warm_up:, model.pfn.nee] * test_Y[k][:, warm_up:, 3], axis=0) / np.sum(test_Y[k][:, warm_up:, 3], axis=0))
    nee_label_test.append(np.sum(test_Y[k][:, warm_up:, 2] * test_Y[k][:, warm_up:, 3], axis=0) / np.sum(test_Y[k][:, warm_up:, 3], axis=0))
    predicted_water_pool_mean = jnp.sum((predicted_sample[:, warm_up:, model.pfn.next_water_pool] ) * test_Y[k][:, warm_up:, 7], axis=1) / (np.sum(test_Y[k][:, warm_up:, 7], axis=1) + 0.1)
    predicted_water_pool_normalized = ((predicted_sample[:, warm_up:, model.pfn.next_water_pool] ).T-predicted_water_pool_mean).T
    predicted_water_pool_normalized = np.mean(predicted_water_pool_normalized * test_Y[k][:, warm_up:, 7], axis=0)
    predicted_water_pool_normalized = jnp.where(jnp.mean(test_Y[k][:, warm_up:, 7], axis=0) < 0.7, jnp.nan, predicted_water_pool_normalized)
    label_water = np.mean(test_Y[k][:, warm_up:, 6] * test_Y[k][:, warm_up:, 7], axis=0)
    label_water = jnp.where(jnp.mean(test_Y[k][:, warm_up:, 7], axis=0) < 0.7, jnp.nan, label_water)
    water_predicted_test.append(predicted_water_pool_normalized)
    water_label_test.append(label_water)
    fire_predicted_test.append(np.sum(predicted_sample[:, warm_up:, model.pfn.total_fire_combust] * test_Y[k][:, warm_up:, 11], axis=0) / np.sum(test_Y[k][:, warm_up:,11], axis=0))
    fire_label_test.append(np.sum(test_Y[k][:, warm_up:, 10] * test_Y[k][:, warm_up:, 11], axis=0) / np.sum(test_Y[k][:, warm_up:, 11], axis=0))
    
nee_predicted_test = np.stack(nee_predicted_test)
nee_label_test = np.stack(nee_label_test)

fire_predicted_test = np.stack(fire_predicted_test)
fire_label_test = np.stack(fire_label_test)

water_predicted_test = np.stack(water_predicted_test)
water_label_test = np.stack(water_label_test)

nee_predicted_dev = []
nee_label_dev = []

water_predicted_dev = []
water_label_dev = []

fire_predicted_dev = []
fire_label_dev = []
for k in range(len(dev_X)):
    predicted_sample = batch_forward(param_state, dev_X[k], dev_MET[k]).squeeze()
    nee_predicted_dev.append(np.sum(predicted_sample[:, warm_up:, model.pfn.nee] * dev_Y[k][:, warm_up:, 3], axis=0) / np.sum(dev_Y[k][:, warm_up:, 3], axis=0))
    nee_label_dev.append(np.sum(dev_Y[k][:, warm_up:, 2] * dev_Y[k][:, warm_up:, 3], axis=0) / np.sum(dev_Y[k][:, warm_up:, 3], axis=0))
    predicted_water_pool_mean = jnp.sum((predicted_sample[:, warm_up:, model.pfn.next_water_pool] ) * dev_Y[k][:, warm_up:, 7], axis=1) / (np.sum(dev_Y[k][:, warm_up:, 7], axis=1) + 0.1)
    predicted_water_pool_normalized = ((predicted_sample[:, warm_up:, model.pfn.next_water_pool] ).T-predicted_water_pool_mean).T
    predicted_water_pool_normalized = np.mean(predicted_water_pool_normalized * dev_Y[k][:, warm_up:, 7], axis=0)
    predicted_water_pool_normalized = jnp.where(jnp.mean(dev_Y[k][:, warm_up:, 7], axis=0) < 0.8, jnp.nan, predicted_water_pool_normalized)
    label_water = np.mean(dev_Y[k][:, warm_up:, 6] * dev_Y[k][:, warm_up:, 7], axis=0)
    label_water = jnp.where(jnp.mean(dev_Y[k][:, warm_up:, 7], axis=0) < 0.8, jnp.nan, label_water)
    water_predicted_dev.append(predicted_water_pool_normalized)
    water_label_dev.append(label_water)
    fire_predicted_dev.append(np.sum(predicted_sample[:, warm_up:, model.pfn.total_fire_combust] * dev_Y[k][:, warm_up:, 11], axis=0) / np.sum(dev_Y[k][:, warm_up:,11], axis=0))
    fire_label_dev.append(np.sum(dev_Y[k][:, warm_up:, 10] * dev_Y[k][:, warm_up:, 11], axis=0) / np.sum(dev_Y[k][:, warm_up:, 11], axis=0))

nee_predicted_dev = np.stack(nee_predicted_dev)
nee_label_dev = np.stack(nee_label_dev)

fire_predicted_dev = np.stack(fire_predicted_dev)
fire_label_dev = np.stack(fire_label_dev)

water_predicted_dev = np.stack(water_predicted_dev)
water_label_dev = np.stack(water_label_dev)

modeled_agb_train = predicted_matrix_train[:, warm_up:, model.pfn.next_labile_pool] 
modeled_agb_train += predicted_matrix_train[:, warm_up:, model.pfn.next_foliar_pool]
modeled_agb_train += predicted_matrix_train[:, warm_up:, model.pfn.next_wood_pool]
modeled_agb_train +=  predicted_matrix_train[:, warm_up:, model.pfn.next_root_pool]
observed_agb_train = output_matrix_train[:, warm_up:, 8]
agb_mask_train = output_matrix_train[:, warm_up:, 9]
    
modeled_agb_train_annual = jnp.mean(modeled_agb_train.reshape(modeled_agb_train.shape[0], -1, 12), axis=2)
observed_agb_train_annual =jnp.mean(observed_agb_train.reshape(observed_agb_train.shape[0], -1, 12), axis=2)
agb_train_mask_annual = jnp.prod(agb_mask_train.reshape(output_matrix_train.shape[0], -1, 12), axis=2)

modeled_agb_test = predicted_matrix_test[:, warm_up:, model.pfn.next_labile_pool] 
modeled_agb_test += predicted_matrix_test[:, warm_up:, model.pfn.next_foliar_pool]
modeled_agb_test += predicted_matrix_test[:, warm_up:, model.pfn.next_wood_pool]
modeled_agb_test +=  predicted_matrix_test[:, warm_up:, model.pfn.next_root_pool]
observed_agb_test = output_matrix_test[:, warm_up:, 8]
agb_mask_test = output_matrix_test[:, warm_up:, 9]
    
modeled_agb_test_annual = jnp.mean(modeled_agb_test.reshape(modeled_agb_test.shape[0], -1, 12), axis=2)
observed_agb_test_annual =jnp.mean(observed_agb_test.reshape(observed_agb_test.shape[0], -1, 12), axis=2)
agb_test_mask_annual = jnp.prod(agb_mask_test.reshape(output_matrix_test.shape[0], -1, 12), axis=2)

modeled_agb_dev = predicted_matrix_dev[:, warm_up:, model.pfn.next_labile_pool] 
modeled_agb_dev += predicted_matrix_dev[:, warm_up:, model.pfn.next_foliar_pool]
modeled_agb_dev += predicted_matrix_dev[:, warm_up:, model.pfn.next_wood_pool]
modeled_agb_dev +=  predicted_matrix_dev[:, warm_up:, model.pfn.next_root_pool]
observed_agb_dev = output_matrix_dev[:, warm_up:, 8]
agb_mask_dev = output_matrix_dev[:, warm_up:, 9]

modeled_agb_dev_annual = jnp.mean(modeled_agb_dev.reshape(modeled_agb_dev.shape[0], -1, 12), axis=2)
observed_agb_dev_annual =jnp.mean(observed_agb_dev.reshape(observed_agb_dev.shape[0], -1, 12), axis=2)
agb_dev_mask_annual = jnp.prod(agb_mask_dev.reshape(output_matrix_dev.shape[0], -1, 12), axis=2)

def compute_metrics(label, predicted, var_name, result_dict):
    invalid = np.isnan(label) | np.isnan(predicted) | (label==-9999)
    label[invalid] = np.nan
    predicted[invalid] = np.nan

    label_flatten = label.flatten()
    predicted_flatten = predicted.flatten()
    sel_flatten = np.invert(np.isnan(label_flatten) | np.isnan(predicted_flatten))
    result_dict[var_name+ "_flat_nse"]=r2_score(label_flatten[sel_flatten], predicted_flatten[sel_flatten])
    result_dict[var_name+ "_flat_mse"]=mean_squared_error(label_flatten[sel_flatten], predicted_flatten[sel_flatten])
    result_dict[var_name+ "_flat_mae"]=mean_absolute_error(label_flatten[sel_flatten], predicted_flatten[sel_flatten]) 
    res = linregress(predicted_flatten[sel_flatten], label_flatten[sel_flatten])
    result_dict[var_name+ "_flat_rval"]=res.rvalue
    result_dict[var_name+ "_flat_slope"]=res.slope
    result_dict[var_name+ "_flat_intercept"]=res.intercept
    result_dict[var_name+ "_flat_pval"]=res.pvalue
    result_dict[var_name+ "_flat_stderr"]=res.stderr
    result_dict[var_name+ "_flat_intercept_stderr"]=res.intercept_stderr
    
    
    label_spatial = np.nanmean(label, axis=1)
    predicted_spatial = np.nanmean(predicted, axis=1)
    sel_spatial = np.invert(np.isnan(label_spatial) | np.isnan(predicted_spatial))

    result_dict[var_name+ "_spatial_nse"]=r2_score(label_spatial[sel_spatial], predicted_spatial[sel_spatial])
    result_dict[var_name+ "_spatial_mse"]=mean_squared_error(label_spatial[sel_spatial], predicted_spatial[sel_spatial])
    result_dict[var_name+ "_spatial_mae"]=mean_absolute_error(label_spatial[sel_spatial], predicted_spatial[sel_spatial]) 
    res = linregress(predicted_spatial[sel_spatial], label_spatial[sel_spatial])
    result_dict[var_name+ "_spatial_rval"]=res.rvalue
    result_dict[var_name+ "_spatial_slope"]=res.slope
    result_dict[var_name+ "_spatial_intercept"]=res.intercept
    result_dict[var_name+ "_spatial_pval"]=res.pvalue
    result_dict[var_name+ "_spatial_stderr"]=res.stderr
    result_dict[var_name+ "_spatial_intercept_stderr"]=res.intercept_stderr
    
    spatial_r2_list = []
    for i in range(label.shape[0]):
        s_sel = np.invert(np.isnan(label[i, :]) | np.isnan(predicted[i, :]))
        if np.sum(s_sel) > 0:
            spatial_r2_list.append(r2_score(label[i, :][s_sel], predicted[i, :][s_sel]))

    result_dict[var_name+ "_spatial_mean_temporal_nse"]=np.nanmean(spatial_r2_list)
    result_dict[var_name+ "_spatial_median_temporal_nse"]=np.nanmedian(spatial_r2_list)
    result_dict[var_name+ "_spatial_std_temporal_nse"]=np.nanstd(spatial_r2_list)
  
    return result_dict

result_dict={}

compute_metrics(np.array(output_matrix_train[:, warm_up:, 4]), 
                np.array(predicted_matrix_train[:, warm_up:, model.pfn.lai]),
                "lai_train", result_dict)

compute_metrics(np.array(output_matrix_train[:, warm_up:, 14]), 
                np.array(predicted_matrix_train[:, warm_up:, model.pfn.vod]),
                "vod_train", result_dict)

compute_metrics(np.array(output_matrix_train[:, warm_up:, 0]), 
                np.array(predicted_matrix_train[:, warm_up:, model.pfn.SIF]),
                "sif_train", result_dict)

compute_metrics(np.array(observed_agb_train_annual), 
                np.array(modeled_agb_train_annual),
                "agb_train", result_dict)

compute_metrics(np.array(output_matrix_train[:, warm_up:, 0]), 
                np.array(predicted_matrix_train[:, warm_up:, model.pfn.SIF]),
                "sif_train", result_dict)

compute_metrics(np.array(output_matrix_train[:, warm_up:, 22]), 
                np.array(predicted_matrix_train[:, warm_up:, model.pfn.ET]),
                "et_train", result_dict)

compute_metrics(nee_label_train, nee_predicted_train, "nee_train", result_dict)
compute_metrics(fire_label_train, fire_predicted_train, "fire_train", result_dict)
compute_metrics(water_label_train, water_predicted_train, "water_train", result_dict)

compute_metrics(np.array(output_matrix_test[:, warm_up:, 4]), 
                np.array(predicted_matrix_test[:, warm_up:, model.pfn.lai]),
                "lai_test", result_dict)

compute_metrics(np.array(output_matrix_test[:, warm_up:, 14]), 
                np.array(predicted_matrix_test[:, warm_up:, model.pfn.vod]),
                "vod_test", result_dict)

compute_metrics(np.array(output_matrix_test[:, warm_up:, 0]), 
                np.array(predicted_matrix_test[:, warm_up:, model.pfn.SIF]),
                "sif_test", result_dict)

compute_metrics(np.array(observed_agb_test_annual), 
                np.array(modeled_agb_test_annual),
                "agb_test", result_dict)

compute_metrics(np.array(output_matrix_test[:, warm_up:, 22]), 
                np.array(predicted_matrix_test[:, warm_up:, model.pfn.ET]),
                "et_test", result_dict)


compute_metrics(nee_label_test, nee_predicted_test, "nee_test", result_dict)
compute_metrics(fire_label_test, fire_predicted_test, "fire_test", result_dict)
compute_metrics(water_label_test, water_predicted_test, "water_test", result_dict)


compute_metrics(np.array(output_matrix_dev[:, warm_up:, 4]), 
                np.array(predicted_matrix_dev[:, warm_up:, model.pfn.lai]),
                "lai_dev", result_dict)

compute_metrics(np.array(output_matrix_dev[:, warm_up:, 14]), 
                np.array(predicted_matrix_dev[:, warm_up:, model.pfn.vod]),
                "vod_dev", result_dict)

compute_metrics(np.array(output_matrix_dev[:, warm_up:, 22]), 
                np.array(predicted_matrix_dev[:, warm_up:, model.pfn.ET]),
                "et_dev", result_dict)


compute_metrics(np.array(output_matrix_dev[:, warm_up:, 0]), 
                np.array(predicted_matrix_dev[:, warm_up:, model.pfn.SIF]),
                "sif_dev", result_dict)


compute_metrics(np.array(observed_agb_dev_annual), 
                np.array(modeled_agb_dev_annual),
                "agb_dev", result_dict)


compute_metrics(nee_label_dev, nee_predicted_dev, "nee_dev", result_dict)

compute_metrics(fire_label_dev, fire_predicted_dev, "fire_dev", result_dict)

compute_metrics(water_label_dev, water_predicted_dev, "water_dev", result_dict)


with open(os.path.join(metrics_dir, "{}_metrics.json".format(exp_str)), "w") as fp:
    json.dump(str(result_dict), fp)

if args.verbose:
    print("Model performance statistics saved to {}".format(os.path.join(metrics_dir, "{}_metrics.json".format(exp_str))))

def evaluate_performance(predicted, true, ax, axis_min, axis_max, title_name=None):
    sel = np.invert(np.isnan(true) | np.isnan(predicted) | (true==-9999))
    true = np.array(true)
    predicted = np.array(predicted)
    true = true[sel]
    predicted = predicted[sel]
    
    res = linregress(predicted, true)

    # histogram the data
    
    PDF=fastkde.pdf(predicted, true, var_names = ['x', 'y'])
    
    hh, locx, locy = PDF.values, PDF.x.values, PDF.y.values

    # Sort the points by density, so that the densest points are plotted last
    z = np.array([hh[np.argmax(b<=locy), np.argmax(a<=locx)] for a,b in zip(predicted,true)])
    idx = z.argsort()
    x2, y2, z2 = predicted[idx], true[idx], z[idx]
    min_positive_z2 = np.min(z2[z2>0])
    
    if np.sum(sel)> 50000:
        s = ax.scatter(x2, y2, c=np.log(np.maximum(z2, min_positive_z2)), cmap='jet', marker='.', s=0.2)
    else:
        s = ax.scatter(x2, y2, c=np.log(np.maximum(z2, min_positive_z2)), cmap='jet', marker='.', s=0.5)
    ax.plot([axis_min, axis_max], [axis_min, axis_max], "k--")
    ax.plot(predicted, res.intercept + res.slope*predicted, 'red', label='fitted line')
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.set_box_aspect(1)
    ax.set_title(title_name, fontsize=14)
    #ax.text(0.1, 0.85, IGBP_name,
    #    verticalalignment='bottom', horizontalalignment='left',
    #    transform=ax.transAxes, fontsize=14)
    ax.tick_params(direction="in")
    metric_dict=dict()
    metric_dict["r2"]=r2_score(true, predicted)
    res = linregress(predicted, true)
    metric_dict["slope"]=res.slope
    metric_dict["intercept"]=res.intercept
    metric_dict["mse"]=mean_squared_error(true, predicted, squared=False)
    
    if metric_dict["intercept"] >= 0:
        ax.text(0.95, 0.26, '{:.3f}x + {:.3f}'.format(metric_dict["slope"], metric_dict["intercept"]),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.95, 0.26, '{:.3f}x {:.3f}'.format(metric_dict["slope"], metric_dict["intercept"]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.19, 'N={0:.{1}f}'.format(true.shape[0], 0),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.12, '$R^2$: {0:.{1}f}'.format(metric_dict["r2"], 2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.05, 'RMSE: {0:.{1}f}'.format(metric_dict["mse"], 2),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=12)
    
fig, axs = plt.subplots(2,4, figsize=(18,9), dpi=300)
ax = axs.flatten()
evaluate_performance(predicted_matrix_dev[:, warm_up:, model.pfn.lai].flatten(), output_matrix_dev[:, warm_up:, 4].flatten(), ax[0], 0, 7.5, "LAI (m$^2$ m$^{-2}$)")
evaluate_performance(predicted_matrix_dev[:, warm_up:, model.pfn.SIF].flatten(), output_matrix_dev[:, warm_up:, 0].flatten(),  ax[1], 0, 0.75, "SIF (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
evaluate_performance(nee_predicted_dev.flatten(), nee_label_dev.flatten(), ax[2], -5,5, "NEE (gC m$^{-2}$ day $^{-1}$)")
evaluate_performance(water_predicted_dev.flatten(), water_label_dev.flatten(), ax[3], -500, 500, "Water Anomaly " + "(kg H$_{2}$O m$^{-2}$)")
evaluate_performance(predicted_matrix_dev[:, warm_up:, model.pfn.vod].flatten(), output_matrix_dev[:, warm_up:, 14].flatten(), ax[4], 0, 1.4, "VOD")
evaluate_performance(modeled_agb_dev_annual.flatten(), observed_agb_dev_annual.flatten(), ax[5], 0, 50000, "Live Biomass (gC m$^{-2}$)")
evaluate_performance(predicted_matrix_dev[:, warm_up:, model.pfn.ET].flatten(), output_matrix_dev[:, warm_up:, 22].flatten(),  ax[6], 0, 7, "Evapotranspiration (mm day $^{-1}$)")
evaluate_performance(fire_predicted_dev.flatten(), fire_label_dev.flatten(),  ax[7], 0, 2.5, "Fire C Emission (gC m$^{-2}$ day #$^{-1}$)")

plt.savefig(os.path.join(fig_dir, "{}_dev.png".format(exp_str)))


fig, axs = plt.subplots(2,4, figsize=(18,9), dpi=300)
ax = axs.flatten()
evaluate_performance(predicted_matrix_train[:, warm_up:, model.pfn.lai].flatten(), output_matrix_train[:, warm_up:, 4].flatten(), ax[0], 0, 7.5, "LAI (m$^2$ m$^{-2}$)")
evaluate_performance(predicted_matrix_train[:, warm_up:, model.pfn.SIF].flatten(), output_matrix_train[:, warm_up:, 0].flatten(),  ax[1], 0, 0.75, "SIF (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
evaluate_performance(nee_predicted_train.flatten(), nee_label_train.flatten(), ax[2], -5,5, "NEE (gC m$^{-2}$ day $^{-1}$)")
evaluate_performance(water_predicted_train.flatten(), water_label_train.flatten(), ax[3], -500, 500, "Water Anomaly " + "(kg H$_{2}$O m$^{-2}$)")
evaluate_performance(predicted_matrix_train[:, warm_up:, model.pfn.vod].flatten(), output_matrix_train[:, warm_up:, 14].flatten(), ax[4], 0, 1.4, "VOD")
evaluate_performance(modeled_agb_train_annual.flatten(), observed_agb_train_annual.flatten(), ax[5], 0, 50000, "Live Biomass (gC m$^{-2}$)")
evaluate_performance(predicted_matrix_train[:, warm_up:, model.pfn.ET].flatten(), output_matrix_train[:, warm_up:, 22].flatten(),  ax[6], 0, 7, "Evapotranspiration (mm day $^{-1}$)")
evaluate_performance(fire_predicted_train.flatten(), fire_label_train.flatten(),  ax[7], 0, 2.5, "Fire C Emission (gC m$^{-2}$ day #$^{-1}$)")
plt.savefig(os.path.join(fig_dir, "{}_train.png".format(exp_str)))

fig, axs = plt.subplots(2,4, figsize=(18,9), dpi=300)
ax = axs.flatten()
evaluate_performance(predicted_matrix_test[:, warm_up:, model.pfn.lai].flatten(), output_matrix_test[:, warm_up:, 4].flatten(), ax[0], 0, 7.5, "LAI (m$^2$ m$^{-2}$)")
evaluate_performance(predicted_matrix_test[:, warm_up:, model.pfn.SIF].flatten(), output_matrix_test[:, warm_up:, 0].flatten(),  ax[1], 0, 0.75, "SIF (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
evaluate_performance(nee_predicted_test.flatten(), nee_label_test.flatten(), ax[2], -5,5, "NEE (gC m$^{-2}$ day $^{-1}$)")
evaluate_performance(water_predicted_test.flatten(), water_label_test.flatten(), ax[3], -500, 500, "Water Anomaly " + "(kg H$_{2}$O m$^{-2}$)")
evaluate_performance(predicted_matrix_test[:, warm_up:, model.pfn.vod].flatten(), output_matrix_test[:, warm_up:, 14].flatten(), ax[4], 0, 1.4, "VOD")
evaluate_performance(modeled_agb_test_annual.flatten(), observed_agb_test_annual.flatten(), ax[5], 0, 50000, "Live Biomass (gC m$^{-2}$)")
evaluate_performance(predicted_matrix_test[:, warm_up:, model.pfn.ET].flatten(), output_matrix_test[:, warm_up:, 22].flatten(),  ax[6], 0, 7, "Evapotranspiration (mm day $^{-1}$)")
evaluate_performance(fire_predicted_test.flatten(), fire_label_test.flatten(),  ax[7], 0, 2.5, "Fire C Emission (gC m$^{-2}$ day #$^{-1}$)")
plt.savefig(os.path.join(fig_dir, "{}_test.png".format(exp_str)))

if args.versbose:
    print("Model performance figures saved to {}".format(fig_dir))

# read in spatial predictors
predictor_matrix_all = read_multiple_varible_to_array(data_dir, "differland_global_driver_v5.4.nc", predictor_list)
# get the CMS-Flux index
RUN_SIMULATION_IDX = read_variable_to_vector(data_dir, "run_simulation_idx_v5.4.nc", "run_simulation_idx", time_idx=run-1)
VALID = read_variable_to_vector(data_dir, "era_valid_v5.4.nc", "era_valid")
INVALID = np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID)
predictor_matrix_all[:, INVALID] = np.nan

scaled_predictor_matrix_all = (predictor_matrix_all.T - np.mean(predictor_matrix_train, axis=1)) / np.std(predictor_matrix_train, axis=1)
all_matrix = jnp.array(scaled_predictor_matrix_all, dtype=jnp.float32)

met_list = ["DAYS", "T_min", "T_max", "SOLR", "CO2", "DOY", "BURNED_AREA", "VPD", "PREC", "LAT", "DELTA_T", "MAT", "MAP"]
met_matrix_all = read_multiple_variable_temporal_to_vector(data_dir, "differland_global_driver_v5.4.nc", met_list, np.full(predictor_matrix_all.shape[1], True), np.arange(0, predictor_matrix_all.shape[1]), n_t=NT)
met_matrix_all = jnp.transpose(met_matrix_all, axes=[2, 1, 0])


def param_forward(params, predictors, met, model):
    embedded_param_state, dalec_param_state, initial_param_state, pheno_param_state = params
    embedded_layer = embed_prediction_forward(embedded_param_state, predictors[1:])
    dalec_parameters = model.unnormalize(parameter_prediction_forward(dalec_param_state, embedded_layer))
    augmented_pheno_predictors = jnp.concatenate([embedded_layer, predictors[0:1]])
    pheno_parameters = model.unnormalize_pheno(parameter_prediction_forward(pheno_param_state, augmented_pheno_predictors))
    initial_pools = model.unnormalize_pools(parameter_prediction_forward(initial_param_state, embedded_layer))

    return embedded_layer, dalec_parameters, pheno_parameters, initial_pools

param_forward = partial(param_forward, model=model)
batch_param_forward = jax.jit(jax.vmap(jax.jit(param_forward), in_axes=[None, 0, 0]))


embedded_layer_all, dalec_parameters_all, pheno_parameters_all, initial_pools_all = batch_param_forward(param_state, all_matrix, met_matrix_all)  
embedded_layer_all=embedded_layer_all.reshape(720, 1440, embedded_layer_all.shape[-1])
dalec_parameters_all=dalec_parameters_all.reshape(720, 1440, dalec_parameters_all.shape[-1])
pheno_parameters_all=pheno_parameters_all.reshape(720, 1440, pheno_parameters_all.shape[-1])
initial_pools_all=initial_pools_all.reshape(720, 1440, initial_pools_all.shape[-1])

dalec_parnames = ["decomposition_rate",
"f_gpp",
"f_fol",
"f_root",
"leaf_lifespan",
"tor_wood",
"tor_root",
"tor_litter",
"tor_som",
"Q10",
"canopy_efficiency",
"flab",
"clab_release_period",
"leaf_fall_period",
"LCMA",
"IWUE",
"runoff_focal_point",
"field_capacity",  
"foliar_cf",
"ligneous_cf",
"dom_cf",
"resilience",
"lab_lifespan",
"moisture_factor",
"uWUE",
"boese_r",
"wilting_point_frac",
"sif_alpha",
"sif_beta_plus_three", "p_fol", "p_wood"]

pheno_parnames = ["Bday", "Fday"]

initial_poolnames = ["Clab0", "Cfol0", "Croot0", "Cwood0", "Clitter0", "Csom0", "Water0"]

ds=xr.Dataset()

for i in range(len(dalec_parnames)):
    ds[dalec_parnames[i]] = xr.DataArray(np.expand_dims(dalec_parameters_all[:, :, i], axis=0), 
                                        coords={"run":[run,],
                                            "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

for i in range(len(pheno_parnames)):
    ds[pheno_parnames[i]] = xr.DataArray(np.expand_dims(pheno_parameters_all[:, :, i], axis=0), 
                                        coords={"run":[run,],
                                            "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

for i in range(len(initial_poolnames)):
    ds[initial_poolnames[i]] = xr.DataArray(np.expand_dims(initial_pools_all[:, :, i], axis=0), 
                                        coords={"run":[run,],
                                            "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})


ds.to_netcdf(os.path.join(nc_dir, "{}_params.nc".format(exp_str)))
if args.versbose:
    print("Maps of ecological parameters saved to {}".format(os.path.join(nc_dir, "{}_params.nc".format(exp_str))))
predicted_all = batch_forward(param_state, all_matrix, met_matrix_all)  
predicted_all = predicted_all.reshape(720, 1440, 168, 43)

def lag_linregress_3D(x, y, lagx=0, lagy=0):
    """
    Input: Two xr.Datarrays of any dimensions with the first dim being time. 
    Thus the input data could be a 1D time series, or for example, have three dimensions (time,lat,lon). 
    Datasets can be provied in any order, but note that the regression slope and intercept will be calculated
    for y with respect to x.
    Output: Covariance, correlation, regression slope and intercept, p-value, and standard error on regression
    between the two datasets along their aligned time dimension.  
    Lag values can be assigned to either of the data, with lagx shifting x, and lagy shifting y, with the specified lag amount.
    Reference: https://hrishichandanpurkar.blogspot.com/2017/09/vectorized-functions-for-correlation.html
    """ 
    #1. Ensure that the data are properly alinged to each other. 
    x,y = xr.align(x,y)


    #2. Add lag information if any, and shift the data accordingly
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards. 
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time')
        #Next important step is to re-align the two datasets so that y adjusts to the changed coordinates of x
        x,y = xr.align(x,y)

    if lagy!=0:
        y   = y.shift(time = -lagy).dropna(dim='time')
        x,y = xr.align(x,y)

    #3. Compute data length, mean and standard deviation along time axis for further use: 
    #n     = x.shape[0]
    n = np.sum(np.invert(np.isnan(x.values)) & np.invert(np.isnan(y.values)), axis=0)
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xstd  = x.std(axis=0)
    ystd  = y.std(axis=0)

    #4. Compute covariance along time axis
    cov   =  np.sum((x - xmean)*(y - ymean), axis=0)/(n)

    #5. Compute correlation along time axis
    cor   = cov/(xstd*ystd)

    #6. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope  

    #7. Compute P-value and standard error
    #Compute t-statistics
    tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
    stderr = slope/tstats

    from scipy.stats import t
    pval   = t.sf(tstats, n-2)*2
    pval   = xr.DataArray(pval, dims=cor.dims, coords=cor.coords)

    return cov,cor,slope,intercept,pval,stderr


observed_lai_da = xr.open_dataset(os.path.join(data_dir, "differland_global_driver_v5.4.nc"))["LAI"][warm_up:, :, :]
predicted_lai_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.lai].transpose([2,0,1]), coords={"time":observed_lai_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_lai, cor_lai, slope_lai, intercept_lai, pval_lai, stderr_lai = lag_linregress_3D(predicted_lai_da, observed_lai_da)


observed_vod_da = xr.open_dataset(os.path.join(data_dir, "differland_global_driver_v5.4.nc"))["VOD"][warm_up:, :, :]
predicted_vod_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.vod].transpose([2,0,1]), coords={"time":observed_vod_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_vod, cor_vod, slope_vod, intercept_vod, pval_vod, stderr_vod = lag_linregress_3D(predicted_vod_da, observed_vod_da)


observed_sif_da = xr.open_dataset(os.path.join(data_dir, "differland_global_driver_v5.4.nc"))["SIF"][warm_up:, :, :]
predicted_sif_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.SIF].transpose([2,0,1]), coords={"time":observed_sif_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_sif, cor_sif, slope_sif, intercept_sif, pval_sif, stderr_sif = lag_linregress_3D(predicted_sif_da, observed_sif_da)

observed_et_da = xr.open_dataset(os.path.join(data_dir, "differland_global_driver_v5.4.nc"))["ET"][warm_up:, :, :]

predicted_et_vals = np.array(predicted_all[:, :, warm_up:, model.pfn.ET].transpose([2,0,1]))
for i in range(NT):
    predicted_et_vals[i, :, :][np.isnan(predicted_sif_da.values[i, :, :])] = np.nan
        
predicted_et_da = xr.DataArray(predicted_et_vals, coords={"time":observed_et_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_et, cor_et, slope_et, intercept_et, pval_et, stderr_et = lag_linregress_3D(predicted_et_da, observed_et_da)


observed_nbe_da = xr.open_dataset(os.path.join(data_dir, "differland_global_driver_v5.4.nc"))["NBE"][warm_up:, :, :]
predicted_nbe_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.nee].transpose([2,0,1]), coords={"time":observed_nbe_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_nbe, cor_nbe, slope_nbe, intercept_nbe, pval_nbe, stderr_nbe = lag_linregress_3D(predicted_nbe_da, observed_nbe_da)

observed_ewt_da = xr.open_dataset(os.path.join(data_dir, "differland_global_driver_v5.4.nc"))["LWE_normalized"][warm_up:, :, :]
predicted_ewt_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.next_water_pool].transpose([2,0,1]), coords={"time":observed_ewt_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_ewt, cor_ewt, slope_ewt, intercept_ewt, pval_ewt, stderr_ewt = lag_linregress_3D(predicted_ewt_da, observed_ewt_da)

fig, axs = plt.subplots(3, 2, figsize=(12,10), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
ax = axs.flatten()

ccmap = LinearSegmentedColormap.from_list("", ["red","whitesmoke", "blue"])

ax[0].imshow(cor_lai, transform=ccrs.PlateCarree(),
        aspect='auto', extent=[-180,180,-90,90],
        interpolation="none", cmap=ccmap, vmin=-1, vmax=1)

ax[0].add_feature(cartopy.feature.OCEAN, zorder=0, color="white")
ax[0].coastlines()
gl0=ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl0.right_labels=False
gl0.top_labels=False
gl0.left_labels=True

ax[0].set_title("LAI", fontsize=14)

ax[1].imshow(cor_sif, transform=ccrs.PlateCarree(),
        aspect='auto', extent=[-180,180,-90,90],
        interpolation="none", cmap=ccmap, vmin=-1, vmax=1)

ax[1].add_feature(cartopy.feature.OCEAN, zorder=0, color="white")
ax[1].coastlines()
gl1=ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl1.right_labels=False
gl1.top_labels=False
gl1.left_labels=True

ax[1].set_title("SIF", fontsize=14)


ax[2].imshow(cor_nbe, transform=ccrs.PlateCarree(),
        aspect='auto', extent=[-180,180,-90,90],
        interpolation="none", cmap=ccmap, vmin=-1, vmax=1)

ax[2].add_feature(cartopy.feature.OCEAN, zorder=0, color="white")
ax[2].coastlines()
gl2=ax[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl2.right_labels=False
gl2.top_labels=False
gl2left_labels=True

ax[2].set_title("NEE", fontsize=14)

ax[3].imshow(cor_ewt, transform=ccrs.PlateCarree(),
        aspect='auto', extent=[-180,180,-90,90],
        interpolation="none", cmap=ccmap, vmin=-1, vmax=1)

ax[3].add_feature(cartopy.feature.OCEAN, zorder=0, color="white")
ax[3].coastlines()
gl3=ax[3].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl3.right_labels=False
gl3.top_labels=False
gl3left_labels=True

ax[3].set_title("EWT", fontsize=14)


ax[4].imshow(cor_et, transform=ccrs.PlateCarree(),
        aspect='auto', extent=[-180,180,-90,90],
        interpolation="none", cmap=ccmap, vmin=-1, vmax=1)

ax[4].add_feature(cartopy.feature.OCEAN, zorder=0, color="white")
ax[4].coastlines()
gl4=ax[4].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl4.right_labels=False
gl4.top_labels=False
gl4.left_labels=True

ax[4].set_title("ET", fontsize=14)


ax[5].imshow(cor_vod, transform=ccrs.PlateCarree(),
        aspect='auto', extent=[-180,180,-90,90],
        interpolation="none", cmap=ccmap, vmin=-1, vmax=1)

ax[5].add_feature(cartopy.feature.OCEAN, zorder=0, color="white")
ax[5].coastlines()
gl5=ax[5].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl5.right_labels=False
gl5.top_labels=False
gl5.left_labels=True

ax[5].set_title("VOD", fontsize=14)


plt.savefig(os.path.join(fig_dir, "{}_spatial_r_val.png".format(exp_str)), dpi=300)

if args.versbose:
    print("Maps of pixelwise correlations between modeled and observed variables saved to {}".format(os.path.join(fig_dir, "{}_spatial_r_val.png".format(exp_str))))

metrics_ds = xr.Dataset()
metrics_ds["cor_nbe"] = cor_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_sif"] = cor_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_lai"] = cor_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_ewt"] = cor_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_et"] = cor_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_vod"] = cor_vod.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds.to_netcdf(os.path.join(nc_dir, "{}_cor.nc".format(exp_str)))

if args.versbose:
    print("Maps of correlation coefficients saved to {}".format(os.path.join(nc_dir, "{}_cor.nc".format(exp_str))))

metrics_ds = xr.Dataset()
metrics_ds["slope_nbe"] = slope_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_sif"] = slope_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_lai"] = slope_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_ewt"] = slope_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_et"] = slope_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_vod"] = slope_vod.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["intercept_nbe"] = intercept_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_sif"] = intercept_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_lai"] = intercept_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_ewt"] = intercept_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_et"] = intercept_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_vod"] = intercept_vod.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["pval_nbe"] = pval_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_sif"] = pval_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_lai"] = pval_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_ewt"] = pval_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_et"] = pval_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_vod"] = pval_vod.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["nbe_mean"] = np.mean(predicted_nbe_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["sif_mean"] = np.mean(predicted_sif_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["ewt_mean"] = np.mean(predicted_ewt_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["lai_mean"] = np.mean(predicted_lai_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["et_mean"] = np.mean(predicted_et_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["vod_mean"] = np.mean(predicted_vod_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)

predicted_beta_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.beta].transpose([2,0,1]), coords={"time":observed_sif_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

metrics_ds["beta_mean"] = np.mean(predicted_beta_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)[0, :, :]
metrics_ds["beta_std"] = np.std(predicted_beta_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)[0, :, :]

predicted_labile_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.next_labile_pool].transpose([2,0,1]), coords={"time":observed_ewt_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

metrics_ds["labile_mean"] = np.mean(predicted_labile_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["labile_initial"] = predicted_labile_da[0, :, :].expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["labile_final"] = predicted_labile_da[-12, :, :].expand_dims(dim={"run":[run,]}, axis=0)


predicted_foliar_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.next_foliar_pool].transpose([2,0,1]), coords={"time":observed_ewt_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

metrics_ds["foliar_mean"] = np.mean(predicted_foliar_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["foliar_initial"] = predicted_foliar_da[0, :, :].expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["foliar_final"] = predicted_foliar_da[-12, :, :].expand_dims(dim={"run":[run,]}, axis=0)


predicted_wood_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.next_wood_pool].transpose([2,0,1]), coords={"time":observed_ewt_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

metrics_ds["wood_mean"] = np.mean(predicted_wood_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["wood_initial"] = predicted_wood_da[0, :, :].expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["wood_final"] = predicted_wood_da[-12, :, :].expand_dims(dim={"run":[run,]}, axis=0)


predicted_root_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.next_root_pool].transpose([2,0,1]), coords={"time":observed_ewt_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

metrics_ds["root_mean"] = np.mean(predicted_root_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["root_initial"] = predicted_root_da[0, :, :].expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["root_final"] = predicted_root_da[-12, :, :].expand_dims(dim={"run":[run,]}, axis=0)

predicted_litter_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.next_litter_pool].transpose([2,0,1]), coords={"time":observed_ewt_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

metrics_ds["litter_mean"] = np.mean(predicted_litter_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["litter_initial"] = predicted_litter_da[0, :, :].expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["litter_final"] = predicted_litter_da[-12, :, :].expand_dims(dim={"run":[run,]}, axis=0)

predicted_som_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.next_som_pool].transpose([2,0,1]), coords={"time":observed_ewt_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

metrics_ds["som_mean"] = np.mean(predicted_som_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["som_initial"] = predicted_som_da[0, :, :].expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["som_final"] = predicted_som_da[-12, :, :].expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds.to_netcdf(os.path.join(nc_dir, "{}_mean_std.nc".format(exp_str)))

if args.versbose:
    print("Mean and std of select pools saved to {}".format(os.path.join(nc_dir, "{}_mean_std.nc".format(exp_str))))


nbe_ds = xr.Dataset({"predicted_nbe":predicted_nbe_da.expand_dims(dim={"run":[run,]}, axis=0)})
nbe_ds.to_netcdf(os.path.join(nc_dir, "{}_nbe.nc".format(exp_str)))

if args.versbose:
    print("Modeled NBE saved to {}".format(os.path.join(nc_dir, "{}_nbe.nc".format(exp_str))))



predicted_fire_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.total_fire_combust].transpose([2,0,1]), coords={"time":observed_nbe_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

predicted_gpp_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.gpp].transpose([2,0,1]), coords={"time":observed_nbe_da.time,
                                                                            "lat":np.linspace(89.875, -89.875, 720), 
                                    "lon":np.linspace(-179.875, 179.875, 1440)})


gpp_et_ds = xr.Dataset({"predicted_gpp":predicted_gpp_da.expand_dims(dim={"run":[run,]}, axis=0),
                            "predicted_et":predicted_et_da.expand_dims(dim={"run":[run,]}, axis=0),
                            "predicted_fire":predicted_fire_da.expand_dims(dim={"run":[run,]}, axis=0)})
gpp_et_ds.to_netcdf(os.path.join(nc_dir, "{}_gpp_et.nc".format(exp_str)))

if args.versbose:
    print("Modeled GPP & ET saved to {}".format(os.path.join(nc_dir, "{}_gpp_et.nc".format(exp_str))))



with open(os.path.join(npy_dir, "{}_embed.npy".format(exp_str)), "wb") as fp:
    np.save(fp, np.array(embedded_layer_all))

if args.versbose:
    print("Spatial embeding saved to {}".format(os.path.join(npy_dir, "{}_embed.npy".format(exp_str))))

print("Post-processing complete")