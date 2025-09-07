import argparse
import jax
import jax.numpy as jnp
from functools import partial
import os
import numpy as np
import pickle
import warnings
import sys
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import xarray as xr
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from copy import deepcopy


sys.path.insert(1, '..')
from DifferLand.util.preprocessing import read_variable_to_vector, read_multiple_varible_to_array
from DifferLand.util.preprocessing import nan_read_multiple_variable_temporal_to_vector
from DifferLand.util.preprocessing import read_multiple_variable_temporal_to_vector
from DifferLand.util.preprocessing import generate_data_loader, generate_input_loader
from DifferLand.optimization.forward import embed_prediction_forward, parameter_prediction_forward
from DifferLand.model.DALEC993 import DALEC993
from DifferLand.util.preprocessing import create_folder_if_not_exists

parser = argparse.ArgumentParser(
                    prog='Evaluate DifferLand Performance.',
                    description='Post-processing DifferLand runs by computing model performance metrics and output select modeled variables.',
                    epilog='By Jianing Fang')

parser.add_argument('-r', '--run', help="Enter an index for the run, used to identify independent calibrations (1-100)", type=int, required=True)
parser.add_argument('-p', '--predictors', required=True)
parser.add_argument('-n', '--neurons', default=32)
parser.add_argument('-x', '--hidden_layers', default=3)
parser.add_argument('-l', '--learning_rate', default=5e-5)
parser.add_argument('-t', '--number_of_timesteps', default=23*12, type=int)
parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("-w", "--warm_up", default=12*2)

# sensitivity experiment setupts
##### Define the target variables to be assimilated into the framework #####

SIF_PROXY_TARGET = "LCSPP"  # Ootions: LCSPP: "LCSPP" |  Not Assimilated: "none"
NBE_INVERSION_TARGET = (
    "NBE"  # options: CMS-Flux: "NBE" | CAMS: "CAMS_NBE" | Not Assimilated: "none"
)
SATELLITE_LAI_TARGET = "LAI"  # options: MODIS LAI: "LAI" | COPERNICUS LAI: "LAI_COPERNICUS" | Not Assimilated: "none"
GRACE_LWE_TARGET = "LWE_normalized"  # options: GRACE EWT Anomaly: LWE_normalized | Not Assimilated: "none"
BIOMASS_TARGET = "biomass_yan"  # options: Annual biomass from Xu et al. 2021 (Sci Adv. ): "biomass_yan" | IB-AGC VOD Derived Annual Biomass Li et al. 2025 Sci. Data): "biomass_ib" | Not Assimilated: "none"
FIRE_EMISSION_TARGET = "GFED_FIRE_EMISSION" # options: GFED Fire C Emission "GFED_FIRE_EMISSION" | CMS Based Fire Emission Inversion: fire_emission  | Not Assimilated: "none"
STATIC_SOIL_TARGET = "som_const" # options: Harmonized World Soil Database (used for the entire period): "som_const" | Not Assimilated: "none"
VOD_TARGET = "none" # options: Not Assimilated: "none" | 
GPP_FLUXNET_TARGET = "gpp_fluxnet" # options: sites whose reported PFT >25% of the containing grid cell : "gpp_fluxnet" | >10%: "gpp_fluxnet_10percent" | >50%: "gpp_fluxnet_50percent" | Not Assimilated: "none"
RECO_FLUXNET_TARGET ="reco_fluxnet" # options: sites whose reported PFT >25% of the containing grid cell : "reco_fluxnet" | >10%: "reco_fluxnet_10percent" | >50%: "reco_fluxnet_50percent" | Not Assimilated: "none"
ET_FLUXNET_TARGET ="et_fluxnet" # options: sites whose reported PFT >25% of the containing grid cell : "et_fluxnet" | >10%: "et_fluxnet_10percent" | >50%: "et_fluxnet_50percent" | Not Assimilated: "none"
GLEAM_ET_TARGET = "ET" # options: GLEAM ET: "ET" | Not Assimilated: "none"

output_list = [
    SIF_PROXY_TARGET,
    NBE_INVERSION_TARGET,
    SATELLITE_LAI_TARGET,
    GRACE_LWE_TARGET,
    BIOMASS_TARGET,
    FIRE_EMISSION_TARGET,
    STATIC_SOIL_TARGET,
    VOD_TARGET,
    GPP_FLUXNET_TARGET,
    RECO_FLUXNET_TARGET,
    ET_FLUXNET_TARGET,
    GLEAM_ET_TARGET,
]

# define directories for accessing data and storing outputs
CARDAMOM_DRIVER_DATA_DIR = "/burg-archive/glab/users/jf3423/data/CARDAMOM_driver_data/global/"
#CARDAMOM_DRIVER_DATA_DIR = "../data/"
CO2_FILENAME = "co2_mm_gl_01_23.csv"
DIFFERLAND_DRIVER_NAME = "combined_global_initial_v6.nc"

OUTPUT_DIR = "./output/"
LOG_DIR = "./log/"
POSTANALYSIS_DIR = "./postanalysis/"

FIG_DIR = os.path.join(POSTANALYSIS_DIR, "figure/")
NC_DIR = os.path.join(POSTANALYSIS_DIR, "nc/")
NPY_DIR = os.path.join(POSTANALYSIS_DIR, "npy/")
METRICS_DIR = os.path.join(POSTANALYSIS_DIR, "metrics/")
args = parser.parse_args()

# parse command line arguments
if args.verbose:
    print("Now start post-processing DifferLand output...")

def get_predictor_list(predictor_set):
    predictor_list = ["LAT_SIGMOID"]
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
LEARNING_RATE = args.learning_rate
NT = args.number_of_timesteps
warm_up = args.warm_up
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
    print("Learning rate: {}".format(LEARNING_RATE))
    print("Number of timesteps: {}".format(NT))
    print("Spatial predictors:")
    
    for p in predictor_list:
        print("\t+ {}".format(p))
        
    print("Targer variables:")
    print(f"\t*{SIF_PROXY_TARGET=}")
    print(f"\t*{NBE_INVERSION_TARGET=}")
    print(f"\t*{SATELLITE_LAI_TARGET=}")
    print(f"\t*{GRACE_LWE_TARGET=}")
    print(f"\t*{BIOMASS_TARGET=}")
    print(f"\t*{FIRE_EMISSION_TARGET=}")
    print(f"\t*{STATIC_SOIL_TARGET=}")
    print(f"\t*{VOD_TARGET=}")
    print(f"\t*{GPP_FLUXNET_TARGET=}")
    print(f"\t*{RECO_FLUXNET_TARGET=}")
    print(f"\t*{ET_FLUXNET_TARGET=}")
    print(f"\t*{GLEAM_ET_TARGET=}")


create_folder_if_not_exists(POSTANALYSIS_DIR, verbose=args.verbose)
create_folder_if_not_exists(FIG_DIR, verbose=args.verbose)
create_folder_if_not_exists(NC_DIR, verbose=args.verbose)
create_folder_if_not_exists(NPY_DIR, verbose=args.verbose)
create_folder_if_not_exists(METRICS_DIR, verbose=args.verbose)



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

def compute_patch_level_nbe(model_name, predictor_loader, met_loader, label_loader, model, warm_up=2*12):
    with open(model_name, "rb") as fp:
        param_state = pickle.load(fp)
        
    nbe_predicted = []
    nbe_label = []
    sif_label = []

    for predictor, met, label in zip(predictor_loader, met_loader, label_loader):
        predicted_sample = batch_forward(param_state, predictor, met).squeeze()
        nbe_predicted.append(np.sum(predicted_sample[:, warm_up:, model.pfn.nbe] * label[:, warm_up:, 3], axis=0) / np.sum(label[:, warm_up:, 3], axis=0))
        nbe_label.append(np.sum(label[:, warm_up:, 2] * label[:, warm_up:, 3], axis=0) / np.sum(label[:, warm_up:, 3], axis=0))
        sif_label.append(np.sum(label[:, warm_up:, 0] * label[:, warm_up:, 1], axis=0) / np.sum(label[:, warm_up:, 1], axis=0))
    
    nbe_predicted_patch = np.stack(nbe_predicted)
    nbe_label_patch = np.stack(nbe_label)
    sif_label_patch = np.stack(sif_label)
    return nbe_predicted_patch, nbe_label_patch, sif_label_patch


def compute_patch_level_water_pool(model_name, predictor_loader, met_loader, label_loader, model, warm_up=2*12):
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

def nan_filtered_mean_absolute_error(targets, predictions):
    sel = np.invert(np.isnan(targets) | np.isnan(predictions))
    if np.sum(sel) > 0:
        return mean_absolute_error(targets[sel], predictions[sel])
    else:
        return np.nan
    
    
def nse_score(targets, predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))
            
def nan_filtered_root_mean_squared_error(y_true, y_pred):
    sel = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(sel) > 0:
        return root_mean_squared_error(y_true[sel], y_pred[sel])
    else:
        return np.nan


def nan_filtered_trend(y_pred):
    #print(y_pred.shape)
    time_x = np.arange(len(y_pred))
    
    sel = ~(np.isnan(y_pred))
    if np.sum(sel) > 0:
        return linregress(time_x[sel], y_pred[sel]).slope
    else:
        return np.nan
    
    
def nan_filtered_trend_p(y_pred):
    time_x = np.arange(len(y_pred))
    
    sel = ~(np.isnan(y_pred))
    if np.sum(sel) > 0:
        return linregress(time_x[sel], y_pred[sel]).pvalue
    else:
        return np.nan

def vectorized_nan_filtered_root_mean_squared_error(y_true, y_pred, dim="time"):
    return xr.apply_ufunc(
        nan_filtered_root_mean_squared_error,
        y_true,
        y_pred,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )

def vectorized_nan_filtered_r2_score(y_true, y_pred, dim="time"):
    return xr.apply_ufunc(
        nan_filtered_r2_score,
        y_true,
        y_pred,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
    )

    
def vectorized_nan_filtered_trend(y_pred, dim="time"):
    return xr.apply_ufunc(
        nan_filtered_trend,
        y_pred.squeeze(),
        input_core_dims=[[dim]],
        vectorize=True,
    )

    
def vectorized_nan_filtered_trend_p(y_pred, dim="time"):
    return xr.apply_ufunc(
        nan_filtered_trend_p,
        y_pred.squeeze(),
        input_core_dims=[[dim]],
        vectorize=True,
    )
    

def detrend_linear(da, dim='time'):
    # Create time axis as float (e.g., years since start)

    time_vals = np.arange(len(da.time))

    # Apply linear detrending across time dimension
    def _detrend(y):
        if np.all(np.isnan(y)):
            return np.full_like(y, np.nan)
        x = time_vals
        mask = ~np.isnan(y)
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        return y - (slope * x + intercept)

    return xr.apply_ufunc(
        _detrend,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype]
    )

def detrend_then_remove_msc(da):
    da_detrend = detrend_linear(da)
    # Remove monthly climatology
    monthly_climatology = da_detrend.groupby("time.month").mean("time")
    da_anom = da_detrend.groupby("time.month") - monthly_climatology
    return da_anom.transpose('time', 'lat', 'lon')
RUN_SIMULATION_IDX = read_variable_to_vector(CARDAMOM_DRIVER_DATA_DIR, "run_simulation_idx_v6.nc", "run_simulation_idx", time_idx=run-1)
    
VALID = read_variable_to_vector(CARDAMOM_DRIVER_DATA_DIR, "era_valid_v6.nc", "era_valid")
INVALID = np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID) | (RUN_SIMULATION_IDX < 0) # filter out dev PIXELS
TEST = np.invert(np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID)) & (RUN_SIMULATION_IDX < 0)
    
predictor_matrix = read_multiple_varible_to_array(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME, predictor_list)
test_predictor_matrix = deepcopy(predictor_matrix)
test_predictor_matrix[:, np.invert(TEST)] = np.nan

predictor_matrix[:, INVALID] = np.nan
ASSIMILATE_BULK_FLAG = read_variable_to_vector(CARDAMOM_DRIVER_DATA_DIR, "assimilate_bulk_variable_v6.nc", "assimilate_bulk_variable")

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
met_matrix = read_multiple_variable_temporal_to_vector(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME, met_list, not_nan_idx, shuffle_idx, n_t=NT)
met_matrix = jnp.transpose(met_matrix, axes=[2, 1, 0])
met_matrix_train = jnp.array(met_matrix[sorted_valid_idx <= train_dev_idx, :], dtype=jnp.float32)
met_matrix_dev = jnp.array(met_matrix[sorted_valid_idx > train_dev_idx, :], dtype=jnp.float32)

test_met_matrix = read_multiple_variable_temporal_to_vector(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME, met_list, test_not_nan_idx, test_shuffle_idx, n_t=NT)
met_matrix_test = jnp.array(jnp.transpose(test_met_matrix, axes=[2, 1, 0]), dtype=jnp.float32)
    
output_matrix = nan_read_multiple_variable_temporal_to_vector(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME, output_list, not_nan_idx, shuffle_idx, n_t=NT)
output_matrix = jnp.transpose(output_matrix, axes=[2, 1, 0])

output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 2].set(-9999)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 3].set(0)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 6].set(-9999)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 7].set(0)
output_matrix_train = jnp.array(output_matrix[sorted_valid_idx <= train_dev_idx, :], dtype=jnp.float32)
output_matrix_dev = jnp.array(output_matrix[sorted_valid_idx > train_dev_idx, :], dtype=jnp.float32)

output_matrix_test = nan_read_multiple_variable_temporal_to_vector(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME, output_list, test_not_nan_idx, test_shuffle_idx, n_t=NT)
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


pickle_name = os.path.join(OUTPUT_DIR, exp_str + ".pickle")

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


nbe_predicted_train = []
nbe_label_train = []

water_predicted_train = []
water_label_train = []

fire_predicted_train = []
fire_label_train = []

for k in range(len(train_X)):
    predicted_sample = batch_forward(param_state, train_X[k], train_MET[k]).squeeze()
    nbe_predicted_train.append(np.sum(predicted_sample[:, warm_up:, model.pfn.nbe] * train_Y[k][:, warm_up:, 3], axis=0) / np.sum(train_Y[k][:, warm_up:, 3], axis=0))
    nbe_label_train.append(np.sum(train_Y[k][:, warm_up:, 2] * train_Y[k][:, warm_up:, 3], axis=0) / np.sum(train_Y[k][:, warm_up:, 3], axis=0))
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
    
nbe_predicted_train = np.stack(nbe_predicted_train)
nbe_label_train = np.stack(nbe_label_train)

fire_predicted_train = np.stack(fire_predicted_train)
fire_label_train = np.stack(fire_label_train)

water_predicted_train = np.stack(water_predicted_train)
water_label_train = np.stack(water_label_train)

nbe_predicted_test = []
nbe_label_test = []

water_predicted_test = []
water_label_test = []

fire_predicted_test = []
fire_label_test = []

for k in range(len(test_X)):
    predicted_sample = batch_forward(param_state, test_X[k], test_MET[k]).squeeze()
    nbe_predicted_test.append(np.sum(predicted_sample[:, warm_up:, model.pfn.nbe] * test_Y[k][:, warm_up:, 3], axis=0) / np.sum(test_Y[k][:, warm_up:, 3], axis=0))
    nbe_label_test.append(np.sum(test_Y[k][:, warm_up:, 2] * test_Y[k][:, warm_up:, 3], axis=0) / np.sum(test_Y[k][:, warm_up:, 3], axis=0))
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
    
nbe_predicted_test = np.stack(nbe_predicted_test)
nbe_label_test = np.stack(nbe_label_test)

fire_predicted_test = np.stack(fire_predicted_test)
fire_label_test = np.stack(fire_label_test)

water_predicted_test = np.stack(water_predicted_test)
water_label_test = np.stack(water_label_test)

nbe_predicted_dev = []
nbe_label_dev = []

water_predicted_dev = []
water_label_dev = []

fire_predicted_dev = []
fire_label_dev = []
for k in range(len(dev_X)):
    predicted_sample = batch_forward(param_state, dev_X[k], dev_MET[k]).squeeze()
    nbe_predicted_dev.append(np.sum(predicted_sample[:, warm_up:, model.pfn.nbe] * dev_Y[k][:, warm_up:, 3], axis=0) / np.sum(dev_Y[k][:, warm_up:, 3], axis=0))
    nbe_label_dev.append(np.sum(dev_Y[k][:, warm_up:, 2] * dev_Y[k][:, warm_up:, 3], axis=0) / np.sum(dev_Y[k][:, warm_up:, 3], axis=0))
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

nbe_predicted_dev = np.stack(nbe_predicted_dev)
nbe_label_dev = np.stack(nbe_label_dev)

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
    result_dict[var_name+ "_flat_nse"]=nan_filtered_r2_score(label_flatten[sel_flatten], predicted_flatten[sel_flatten])
    result_dict[var_name+ "_flat_rmse"]=nan_filtered_root_mean_squared_error(label_flatten[sel_flatten], predicted_flatten[sel_flatten])
    result_dict[var_name+ "_flat_mae"]=nan_filtered_mean_absolute_error(label_flatten[sel_flatten], predicted_flatten[sel_flatten]) 

    if np.sum(sel_flatten) > 0:
        res = linregress(predicted_flatten[sel_flatten], label_flatten[sel_flatten])
        result_dict[var_name+ "_flat_rval"]=res.rvalue
        result_dict[var_name+ "_flat_slope"]=res.slope
        result_dict[var_name+ "_flat_intercept"]=res.intercept
        result_dict[var_name+ "_flat_pval"]=res.pvalue
        result_dict[var_name+ "_flat_stderr"]=res.stderr
        result_dict[var_name+ "_flat_intercept_stderr"]=res.intercept_stderr
    else:
        result_dict[var_name+ "_flat_rval"]=np.nan
        result_dict[var_name+ "_flat_slope"]=np.nan
        result_dict[var_name+ "_flat_intercept"]=np.nan
        result_dict[var_name+ "_flat_pval"]=np.nan
        result_dict[var_name+ "_flat_stderr"]=np.nan
        result_dict[var_name+ "_flat_intercept_stderr"]=np.nan
    
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        label_spatial = np.nanmean(label, axis=1)
        predicted_spatial = np.nanmean(predicted, axis=1)
    sel_spatial = np.invert(np.isnan(label_spatial) | np.isnan(predicted_spatial))

    result_dict[var_name+ "_spatial_nse"]=nan_filtered_r2_score(label_spatial[sel_spatial], predicted_spatial[sel_spatial])
    result_dict[var_name+ "_spatial_rmse"]=nan_filtered_root_mean_squared_error(label_spatial[sel_spatial], predicted_spatial[sel_spatial])
    result_dict[var_name+ "_spatial_mae"]=nan_filtered_mean_absolute_error(label_spatial[sel_spatial], predicted_spatial[sel_spatial]) 

    if np.sum(sel_spatial) > 0:
        res = linregress(predicted_spatial[sel_spatial], label_spatial[sel_spatial])
        result_dict[var_name+ "_spatial_rval"]=res.rvalue
        result_dict[var_name+ "_spatial_slope"]=res.slope
        result_dict[var_name+ "_spatial_intercept"]=res.intercept
        result_dict[var_name+ "_spatial_pval"]=res.pvalue
        result_dict[var_name+ "_spatial_stderr"]=res.stderr
        result_dict[var_name+ "_spatial_intercept_stderr"]=res.intercept_stderr
    else:
        result_dict[var_name+ "_spatial_rval"]=np.nan
        result_dict[var_name+ "_spatial_slope"]=np.nan
        result_dict[var_name+ "_spatial_intercept"]=np.nan
        result_dict[var_name+ "_spatial_pval"]=np.nan
        result_dict[var_name+ "_spatial_stderr"]=np.nan
        result_dict[var_name+ "_spatial_intercept_stderr"]=np.nan

    spatial_r2_list = []
    for i in range(label.shape[0]):
        s_sel = np.invert(np.isnan(label[i, :]) | np.isnan(predicted[i, :]))
        if np.sum(s_sel) > 0:
            spatial_r2_list.append(r2_score(label[i, :][s_sel], predicted[i, :][s_sel]))
    spatial_rmse_list = []
    for i in range(label.shape[0]):
        s_sel = np.invert(np.isnan(label[i, :]) | np.isnan(predicted[i, :]))
        if np.sum(s_sel) > 0:
            spatial_rmse_list.append(root_mean_squared_error(label[i, :][s_sel], predicted[i, :][s_sel]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result_dict[var_name+ "_spatial_mean_temporal_nse"]=np.nanmean(spatial_r2_list)
        result_dict[var_name+ "_spatial_median_temporal_nse"]=np.nanmedian(spatial_r2_list)
        result_dict[var_name+ "_spatial_std_temporal_nse"]=np.nanstd(spatial_r2_list)
        
        result_dict[var_name+ "_spatial_mean_temporal_rmse"]=np.nanmean(spatial_rmse_list)
        result_dict[var_name+ "_spatial_median_temporal_rmse"]=np.nanmedian(spatial_rmse_list)
        result_dict[var_name+ "_spatial_std_temporal_rmse"]=np.nanstd(spatial_rmse_list)
  
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

compute_metrics(nbe_label_train, nbe_predicted_train, "nbe_train", result_dict)
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


compute_metrics(nbe_label_test, nbe_predicted_test, "nbe_test", result_dict)
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


compute_metrics(nbe_label_dev, nbe_predicted_dev, "nbe_dev", result_dict)

compute_metrics(fire_label_dev, fire_predicted_dev, "fire_dev", result_dict)

compute_metrics(water_label_dev, water_predicted_dev, "water_dev", result_dict)


with open(os.path.join(METRICS_DIR, "{}_metrics.json".format(exp_str)), "w") as fp:
    json.dump(str(result_dict), fp)

if args.verbose:
    print("Model performance statistics saved to {}".format(os.path.join(METRICS_DIR, "{}_metrics.json".format(exp_str))))

def evaluate_performance(predicted, true, ax, axis_min, axis_max, title_name=None):
    sel = np.invert(np.isnan(true) | np.isnan(predicted) | (true==-9999))
    if np.sum(sel) > 0:
        true = np.array(true)
        predicted = np.array(predicted)
        true = true[sel]
        predicted = predicted[sel]
        
        res = linregress(predicted, true)

        # histogram the data
        
        

        # Sort the points by density, so that the densest points are plotted last
        if np.sum(sel)> 50000:
            ax.scatter(predicted, true, marker='.', s=0.2)
        else:
            ax.scatter(predicted, true, marker='.', s=0.5)
        ax.plot([axis_min, axis_max], [axis_min, axis_max], "k--")
        ax.plot(predicted, res.intercept + res.slope*predicted, 'red', label='fitted line')
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        ax.set_box_aspect(1)
        ax.set_title(title_name, fontsize=14)
        ax.tick_params(direction="in")
        metric_dict=dict()
        metric_dict["r2"]=nan_filtered_r2_score(true, predicted)
        res = linregress(predicted, true)
        metric_dict["slope"]=res.slope
        metric_dict["intercept"]=res.intercept
        metric_dict["rmse"]=nan_filtered_root_mean_squared_error(true, predicted)
        
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
        ax.text(0.95, 0.05, 'RMSE: {0:.{1}f}'.format(metric_dict["rmse"], 2),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontsize=12)
    
fig, axs = plt.subplots(2,4, figsize=(18,9), dpi=300)
ax = axs.flatten()
evaluate_performance(predicted_matrix_dev[:, warm_up:, model.pfn.lai].flatten(), output_matrix_dev[:, warm_up:, 4].flatten(), ax[0], 0, 7.5, "LAI (m$^2$ m$^{-2}$)")
evaluate_performance(predicted_matrix_dev[:, warm_up:, model.pfn.SIF].flatten(), output_matrix_dev[:, warm_up:, 0].flatten(),  ax[1], 0, 0.75, "SIF (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
evaluate_performance(nbe_predicted_dev.flatten(), nbe_label_dev.flatten(), ax[2], -5,5, "NBE (gC m$^{-2}$ day $^{-1}$)")
evaluate_performance(water_predicted_dev.flatten(), water_label_dev.flatten(), ax[3], -500, 500, "Water Anomaly " + "(kg H$_{2}$O m$^{-2}$)")
evaluate_performance(predicted_matrix_dev[:, warm_up:, model.pfn.vod].flatten(), output_matrix_dev[:, warm_up:, 14].flatten(), ax[4], 0, 1.4, "VOD")
evaluate_performance(modeled_agb_dev_annual.flatten(), observed_agb_dev_annual.flatten(), ax[5], 0, 50000, "Live Biomass (gC m$^{-2}$)")
evaluate_performance(predicted_matrix_dev[:, warm_up:, model.pfn.ET].flatten(), output_matrix_dev[:, warm_up:, 22].flatten(),  ax[6], 0, 7, "Evapotranspiration (mm day $^{-1}$)")
evaluate_performance(fire_predicted_dev.flatten(), fire_label_dev.flatten(),  ax[7], 0, 2.5, "Fire C Emission (gC m$^{-2}$ day #$^{-1}$)")

plt.savefig(os.path.join(FIG_DIR, "{}_dev.png".format(exp_str)))


fig, axs = plt.subplots(2,4, figsize=(18,9), dpi=300)
ax = axs.flatten()
evaluate_performance(predicted_matrix_train[:, warm_up:, model.pfn.lai].flatten(), output_matrix_train[:, warm_up:, 4].flatten(), ax[0], 0, 7.5, "LAI (m$^2$ m$^{-2}$)")
evaluate_performance(predicted_matrix_train[:, warm_up:, model.pfn.SIF].flatten(), output_matrix_train[:, warm_up:, 0].flatten(),  ax[1], 0, 0.75, "SIF (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
evaluate_performance(nbe_predicted_train.flatten(), nbe_label_train.flatten(), ax[2], -5,5, "NBE (gC m$^{-2}$ day $^{-1}$)")
evaluate_performance(water_predicted_train.flatten(), water_label_train.flatten(), ax[3], -500, 500, "Water Anomaly " + "(kg H$_{2}$O m$^{-2}$)")
evaluate_performance(predicted_matrix_train[:, warm_up:, model.pfn.vod].flatten(), output_matrix_train[:, warm_up:, 14].flatten(), ax[4], 0, 1.4, "VOD")
evaluate_performance(modeled_agb_train_annual.flatten(), observed_agb_train_annual.flatten(), ax[5], 0, 50000, "Live Biomass (gC m$^{-2}$)")
evaluate_performance(predicted_matrix_train[:, warm_up:, model.pfn.ET].flatten(), output_matrix_train[:, warm_up:, 22].flatten(),  ax[6], 0, 7, "Evapotranspiration (mm day $^{-1}$)")
evaluate_performance(fire_predicted_train.flatten(), fire_label_train.flatten(),  ax[7], 0, 2.5, "Fire C Emission (gC m$^{-2}$ day #$^{-1}$)")
plt.savefig(os.path.join(FIG_DIR, "{}_train.png".format(exp_str)))

fig, axs = plt.subplots(2,4, figsize=(18,9), dpi=300)
ax = axs.flatten()
evaluate_performance(predicted_matrix_test[:, warm_up:, model.pfn.lai].flatten(), output_matrix_test[:, warm_up:, 4].flatten(), ax[0], 0, 7.5, "LAI (m$^2$ m$^{-2}$)")
evaluate_performance(predicted_matrix_test[:, warm_up:, model.pfn.SIF].flatten(), output_matrix_test[:, warm_up:, 0].flatten(),  ax[1], 0, 0.75, "SIF (mW m$^{-2}$ nm$^{-1}$ sr$^{-1}$)")
evaluate_performance(nbe_predicted_test.flatten(), nbe_label_test.flatten(), ax[2], -5,5, "NBE (gC m$^{-2}$ day $^{-1}$)")
evaluate_performance(water_predicted_test.flatten(), water_label_test.flatten(), ax[3], -500, 500, "Water Anomaly " + "(kg H$_{2}$O m$^{-2}$)")
evaluate_performance(predicted_matrix_test[:, warm_up:, model.pfn.vod].flatten(), output_matrix_test[:, warm_up:, 14].flatten(), ax[4], 0, 1.4, "VOD")
evaluate_performance(modeled_agb_test_annual.flatten(), observed_agb_test_annual.flatten(), ax[5], 0, 50000, "Live Biomass (gC m$^{-2}$)")
evaluate_performance(predicted_matrix_test[:, warm_up:, model.pfn.ET].flatten(), output_matrix_test[:, warm_up:, 22].flatten(),  ax[6], 0, 7, "Evapotranspiration (mm day $^{-1}$)")
evaluate_performance(fire_predicted_test.flatten(), fire_label_test.flatten(),  ax[7], 0, 2.5, "Fire C Emission (gC m$^{-2}$ day #$^{-1}$)")
plt.savefig(os.path.join(FIG_DIR, "{}_test.png".format(exp_str)))

if args.verbose:
    print("Model performance figures saved to {}".format(FIG_DIR))

# read in spatial predictors
predictor_matrix_all = read_multiple_varible_to_array(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME, predictor_list)
# get the CMS-Flux index
RUN_SIMULATION_IDX = read_variable_to_vector(CARDAMOM_DRIVER_DATA_DIR, "run_simulation_idx_v6.nc", "run_simulation_idx", time_idx=run-1)
VALID = read_variable_to_vector(CARDAMOM_DRIVER_DATA_DIR, "era_valid_v6.nc", "era_valid")
INVALID = np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID)
predictor_matrix_all[:, INVALID] = np.nan

scaled_predictor_matrix_all = (predictor_matrix_all.T - np.mean(predictor_matrix_train, axis=1)) / np.std(predictor_matrix_train, axis=1)
all_matrix = jnp.array(scaled_predictor_matrix_all, dtype=jnp.float32)

met_list = ["DAYS", "T_min", "T_max", "SOLR", "CO2", "DOY", "BURNED_AREA", "VPD", "PREC", "LAT", "DELTA_T", "MAT", "MAP"]
met_matrix_all = read_multiple_variable_temporal_to_vector(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME, met_list, np.full(predictor_matrix_all.shape[1], True), np.arange(0, predictor_matrix_all.shape[1]), n_t=NT)
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


ds.to_netcdf(os.path.join(NC_DIR, "{}_params.nc".format(exp_str)))
if args.versbose:
    print("Maps of ecological parameters saved to {}".format(os.path.join(NC_DIR, "{}_params.nc".format(exp_str))))
predicted_all = batch_forward(param_state, all_matrix, met_matrix_all)  
predicted_all = predicted_all.reshape(720, 1440, args.number_of_timesteps, 44)

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


observed_lai_da = xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["LAI"][warm_up:, :, :]
predicted_lai_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.lai].transpose([2,0,1]), coords={"time":observed_lai_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_lai, cor_lai, slope_lai, intercept_lai, pval_lai, stderr_lai = lag_linregress_3D(predicted_lai_da, observed_lai_da)

observed_lai_annual_da = observed_lai_da.resample({'time': '1Y'}).mean()
predicted_lai_annual_da = predicted_lai_da.resample({'time': '1Y'}).mean()

cov_lai_annual, cor_lai_annual, slope_lai_annual, intercept_lai_annual, pval_lai_annual, stderr_lai_annual = lag_linregress_3D(predicted_lai_annual_da, observed_lai_annual_da)

observed_lai_anom_da = detrend_then_remove_msc(observed_lai_da)
predicted_lai_anom_da = detrend_then_remove_msc(predicted_lai_da)
cov_lai_anom, cor_lai_anom, slope_lai_anom, intercept_lai_anom, pval_lai_anom, stderr_lai_anom = lag_linregress_3D(predicted_lai_anom_da, observed_lai_anom_da)


observed_sif_da = xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["SIF"][warm_up:, :, :]
predicted_sif_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.SIF].transpose([2,0,1]), coords={"time":observed_sif_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})


observed_sif_annual_da = observed_sif_da.resample({'time': '1Y'}).mean()
predicted_sif_annual_da = predicted_sif_da.resample({'time': '1Y'}).mean()
cov_sif, cor_sif, slope_sif, intercept_sif, pval_sif, stderr_sif = lag_linregress_3D(predicted_sif_da, observed_sif_da)
cov_sif_annual, cor_sif_annual, slope_sif_annual, intercept_sif_annual, pval_sif_annual, stderr_sif_annual = lag_linregress_3D(predicted_sif_annual_da, observed_sif_annual_da)

observed_sif_anom_da = detrend_then_remove_msc(observed_sif_da)
predicted_sif_anom_da = detrend_then_remove_msc(predicted_sif_da)
cov_sif_anom, cor_sif_anom, slope_sif_anom, intercept_sif_anom, pval_sif_anom, stderr_sif_anom = lag_linregress_3D(predicted_sif_anom_da, observed_sif_anom_da)

observed_et_da = xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["ET"][warm_up:, :, :]

predicted_et_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.ET].transpose([2,0,1]), coords={"time":observed_et_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_et, cor_et, slope_et, intercept_et, pval_et, stderr_et = lag_linregress_3D(predicted_et_da, observed_et_da)

observed_et_annual_da = observed_et_da.resample({'time': '1Y'}).mean()
predicted_et_annual_da = predicted_et_da.resample({'time': '1Y'}).mean()
cov_et, cor_et, slope_et, intercept_et, pval_et, stderr_et = lag_linregress_3D(predicted_et_da, observed_et_da)
cov_et_annual, cor_et_annual, slope_et_annual, intercept_et_annual, pval_et_annual, stderr_et_annual = lag_linregress_3D(predicted_et_annual_da, observed_et_annual_da)

observed_et_anom_da = detrend_then_remove_msc(observed_et_da)
predicted_et_anom_da = detrend_then_remove_msc(predicted_et_da)
cov_et_anom, cor_et_anom, slope_et_anom, intercept_et_anom, pval_et_anom, stderr_et_anom = lag_linregress_3D(predicted_et_anom_da, observed_et_anom_da)

def coarsen_numpy(x, block_h=16, block_w=20):
    n, h, w = x.shape
    assert h % block_h == 0 and w % block_w == 0, "Dimensions must be divisible by block size"

    x = x.reshape(n, h // block_h, block_h, w // block_w, block_w)
    x_coarse = np.nanmean(x, axis=(2, 4))  # (n, 16, 20)
    x_up = np.repeat(np.repeat(x_coarse, block_h, axis=1), block_w, axis=2)  # (n, 720, 1440)
    return x_up

observed_nbe_da = xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["NBE"][warm_up:, :, :]
observed_nbe_da.values = coarsen_numpy(observed_nbe_da.values)
predicted_nbe_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.nbe].transpose([2,0,1]), coords={"time":observed_nbe_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})
predicted_nbe_da.values = coarsen_numpy(predicted_nbe_da.values)

cov_nbe, cor_nbe, slope_nbe, intercept_nbe, pval_nbe, stderr_nbe = lag_linregress_3D(predicted_nbe_da, observed_nbe_da)

observed_nbe_annual_da = observed_nbe_da.resample({'time': '1Y'}).mean()
predicted_nbe_annual_da = predicted_nbe_da.resample({'time': '1Y'}).mean()
cov_nbe, cor_nbe, slope_nbe, intercept_nbe, pval_nbe, stderr_nbe = lag_linregress_3D(predicted_nbe_da, observed_nbe_da)
cov_nbe_annual, cor_nbe_annual, slope_nbe_annual, intercept_nbe_annual, pval_nbe_annual, stderr_nbe_annual = lag_linregress_3D(predicted_nbe_annual_da, observed_nbe_annual_da)

observed_ewt_da = xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["LWE_normalized"][warm_up:, :, :]
predicted_ewt_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.next_water_pool].transpose([2,0,1]), coords={"time":observed_ewt_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_ewt, cor_ewt, slope_ewt, intercept_ewt, pval_ewt, stderr_ewt = lag_linregress_3D(predicted_ewt_da, observed_ewt_da)

observed_ewt_annual_da = observed_ewt_da.resample({'time': '1Y'}).mean()
predicted_ewt_annual_da = predicted_ewt_da.resample({'time': '1Y'}).mean()
cov_ewt, cor_ewt, slope_ewt, intercept_ewt, pval_ewt, stderr_ewt = lag_linregress_3D(predicted_ewt_da, observed_ewt_da)
cov_ewt_annual, cor_ewt_annual, slope_ewt_annual, intercept_ewt_annual, pval_ewt_annual, stderr_ewt_annual = lag_linregress_3D(predicted_ewt_annual_da, observed_ewt_annual_da)

if args.biomass == "biomass_ib":
    observed_biomass_da = xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["IB_AGC"][warm_up:, :, :] 
    + xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["IB_BGC"][warm_up:, :, :]
else:
    observed_biomass_da = xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["agb_yan"][warm_up:, :, :] 
    + xr.open_dataset(os.path.join(CARDAMOM_DRIVER_DATA_DIR, DIFFERLAND_DRIVER_NAME))["bgb_yan"][warm_up:, :, :]

predicted_biomass_da = xr.DataArray((predicted_all[:, :, warm_up:, model.pfn.next_foliar_pool]+predicted_all[:, :, warm_up:, model.pfn.next_wood_pool]+predicted_all[:, :, warm_up:, model.pfn.next_labile_pool]+predicted_all[:, :, warm_up:, model.pfn.next_root_pool]).transpose([2,0,1]), coords={"time":observed_biomass_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

cov_biomass, cor_biomass, slope_biomass, intercept_biomass, pval_biomass, stderr_biomass = lag_linregress_3D(predicted_biomass_da, observed_biomass_da)

observed_biomass_annual_da = observed_biomass_da.resample({'time': '1Y'}).mean()
predicted_biomass_annual_da = predicted_biomass_da.resample({'time': '1Y'}).mean()
cov_biomass, cor_biomass, slope_biomass, intercept_biomass, pval_biomass, stderr_biomass = lag_linregress_3D(predicted_biomass_da, observed_biomass_da)
cov_biomass_annual, cor_biomass_annual, slope_biomass_annual, intercept_biomass_annual, pval_biomass_annual, stderr_biomass_annual = lag_linregress_3D(predicted_biomass_annual_da, observed_biomass_annual_da)

rmse_lai = vectorized_nan_filtered_root_mean_squared_error(predicted_lai_da, observed_lai_da)
rmse_sif = vectorized_nan_filtered_root_mean_squared_error(predicted_sif_da, observed_sif_da)
rmse_et = vectorized_nan_filtered_root_mean_squared_error(predicted_et_da, observed_et_da)
rmse_nbe = vectorized_nan_filtered_root_mean_squared_error(predicted_nbe_da, observed_nbe_da)
rmse_ewt = vectorized_nan_filtered_root_mean_squared_error(predicted_ewt_da, observed_ewt_da)
rmse_biomass = vectorized_nan_filtered_root_mean_squared_error(predicted_biomass_da, observed_biomass_da)


r2_lai = vectorized_nan_filtered_r2_score(predicted_lai_da, observed_lai_da)
r2_sif = vectorized_nan_filtered_r2_score(predicted_sif_da, observed_sif_da)
r2_et = vectorized_nan_filtered_r2_score(predicted_et_da, observed_et_da)
r2_nbe = vectorized_nan_filtered_r2_score(predicted_nbe_da, observed_nbe_da)
r2_ewt = vectorized_nan_filtered_r2_score(predicted_ewt_da, observed_ewt_da)
r2_biomass = vectorized_nan_filtered_r2_score(predicted_biomass_da, observed_biomass_da)


trend_lai = vectorized_nan_filtered_trend(predicted_lai_da)
trend_sif = vectorized_nan_filtered_trend(predicted_sif_da)
trend_et = vectorized_nan_filtered_trend(predicted_et_da)
trend_nbe = vectorized_nan_filtered_trend(predicted_nbe_da)
trend_ewt = vectorized_nan_filtered_trend(predicted_ewt_da)
trend_biomass = vectorized_nan_filtered_trend(predicted_biomass_da)

trend_p_lai = vectorized_nan_filtered_trend_p(predicted_lai_da)
trend_p_sif = vectorized_nan_filtered_trend_p(predicted_sif_da)
trend_p_et = vectorized_nan_filtered_trend_p(predicted_et_da)
trend_p_nbe = vectorized_nan_filtered_trend_p(predicted_nbe_da)
trend_p_ewt = vectorized_nan_filtered_trend_p(predicted_ewt_da)
trend_p_biomass = vectorized_nan_filtered_trend_p(predicted_biomass_da)

metrics_ds = xr.Dataset()
metrics_ds["cor_nbe"] = cor_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_sif"] = cor_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_lai"] = cor_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_ewt"] = cor_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_et"] = cor_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_biomass"] = cor_biomass.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["cor_nbe_annual"] = cor_nbe_annual.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_sif_annual"] = cor_sif_annual.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_lai_annual"] = cor_lai_annual.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_ewt_annual"] = cor_ewt_annual.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_et_annual"] = cor_et_annual.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_biomass_annual"] = cor_biomass_annual.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["cor_sif_anom"] = cor_sif_annual.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_lai_anom"] = cor_lai_annual.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["cor_et_anom"] = cor_et_anom.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["rmse_nbe"] = rmse_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["rmse_sif"] = rmse_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["rmse_lai"] = rmse_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["rmse_ewt"] = rmse_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["rmse_et"] = rmse_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["rmse_biomass"] = rmse_biomass.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["r2_nbe"] = r2_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["r2_sif"] = r2_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["r2_lai"] = r2_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["r2_ewt"] = r2_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["r2_et"] = r2_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["r2_biomass"] = r2_biomass.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["trend_nbe"] = trend_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_sif"] = trend_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_lai"] = trend_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_ewt"] = trend_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_et"] = trend_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_biomass"] = trend_biomass.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["trend_p_nbe"] = trend_p_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_p_sif"] = trend_p_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_p_lai"] = trend_p_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_p_ewt"] = trend_p_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_p_et"] = trend_p_et.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["trend_p_biomass"] = trend_p_biomass.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds.to_netcdf(os.path.join("./postanalysis/nc/", "{}_cor.nc".format(exp_str)))

if args.versbose:
    print("Maps of correlation coefficients saved to {}".format(os.path.join(NC_DIR, "{}_cor.nc".format(exp_str))))

metrics_ds = xr.Dataset()
metrics_ds["slope_nbe"] = slope_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_sif"] = slope_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_lai"] = slope_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_ewt"] = slope_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["slope_et"] = slope_et.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["intercept_nbe"] = intercept_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_sif"] = intercept_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_lai"] = intercept_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_ewt"] = intercept_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["intercept_et"] = intercept_et.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["pval_nbe"] = pval_nbe.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_sif"] = pval_sif.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_lai"] = pval_lai.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_ewt"] = pval_ewt.expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["pval_et"] = pval_et.expand_dims(dim={"run":[run,]}, axis=0)

metrics_ds["nbe_mean"] = np.mean(predicted_nbe_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["sif_mean"] = np.mean(predicted_sif_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["ewt_mean"] = np.mean(predicted_ewt_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["lai_mean"] = np.mean(predicted_lai_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)
metrics_ds["et_mean"] = np.mean(predicted_et_da, axis=0).expand_dims(dim={"run":[run,]}, axis=0)

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

metrics_ds.to_netcdf(os.path.join(NC_DIR, "{}_mean_std.nc".format(exp_str)))

if args.verbose:
    print("Mean and std of select pools saved to {}".format(os.path.join(NC_DIR, "{}_mean_std.nc".format(exp_str))))


nbe_ds = xr.Dataset({"predicted_nbe":predicted_nbe_da.expand_dims(dim={"run":[run,]}, axis=0)})
nbe_ds.to_netcdf(os.path.join(NC_DIR, "{}_nbe.nc".format(exp_str)))

if args.verbose:
    print("Modeled NBE saved to {}".format(os.path.join(NC_DIR, "{}_nbe.nc".format(exp_str))))



predicted_fire_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.total_fire_combust].transpose([2,0,1]), coords={"time":observed_nbe_da.time,
                                                                                "lat":np.linspace(89.875, -89.875, 720), 
                                        "lon":np.linspace(-179.875, 179.875, 1440)})

predicted_gpp_da = xr.DataArray(predicted_all[:, :, warm_up:, model.pfn.gpp].transpose([2,0,1]), coords={"time":observed_nbe_da.time,
                                                                            "lat":np.linspace(89.875, -89.875, 720), 
                                    "lon":np.linspace(-179.875, 179.875, 1440)})


gpp_et_ds = xr.Dataset({"predicted_gpp":predicted_gpp_da.expand_dims(dim={"run":[run,]}, axis=0),
                            "predicted_et":predicted_et_da.expand_dims(dim={"run":[run,]}, axis=0),
                            "predicted_fire":predicted_fire_da.expand_dims(dim={"run":[run,]}, axis=0)})
gpp_et_ds.to_netcdf(os.path.join(NC_DIR, "{}_gpp_et.nc".format(exp_str)))

if args.verbose:
    print("Modeled GPP & ET saved to {}".format(os.path.join(NC_DIR, "{}_gpp_et.nc".format(exp_str))))



with open(os.path.join(NPY_DIR, "{}_embed.npy".format(exp_str)), "wb") as fp:
    np.save(fp, np.array(embedded_layer_all))

if args.verbose:
    print("Spatial embeding saved to {}".format(os.path.join(NPY_DIR, "{}_embed.npy".format(exp_str))))

print("Post-processing complete")