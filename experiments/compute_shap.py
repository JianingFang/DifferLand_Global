import argparse
import jax
import jax.numpy as jnp
from functools import partial
import os
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.insert(1, '..')
from DifferLand.util.preprocessing import read_variable_to_vector, read_multiple_varible_to_array
from DifferLand.optimization.forward import embed_prediction_forward, parameter_prediction_forward
from DifferLand.optimization.loss_functions import *

from DifferLand.model.DALEC993 import DALEC993
import xarray as xr
import shap
from DifferLand.util.preprocessing import create_folder_if_not_exists
parser = argparse.ArgumentParser(
                    prog='Compute SHAP values for the spatilization network.',
                    description='JAX implementation of differentiable CARDAMOM!',
                    epilog='By Jianing Fang')

parser.add_argument('-r', '--run', help="Enter an index for the run, used to identify independent calibrations (1-100)", type=int, required=True)
parser.add_argument('-p', '--predictors', required=True)
parser.add_argument('-n', '--neurons', default=32)
parser.add_argument('-x', '--hidden_layers', default=3)
parser.add_argument('-i', '--iterations', default=199, type=int)
parser.add_argument('-l', '--learning_rate', default=5e-5)
parser.add_argument('-t', '--number_of_timesteps', default=168, type=int)
parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("-a", "--normalize", action=argparse.BooleanOptionalAction, default=False)


output_dir = "./output/"
log_dir = "./log/"
postanalysis_dir = "./postanalysis/"

nc_dir = "./postanalysis/nc/"
shap_dir = "./postanalysis/shap/"
data_dir = "../data/"
args = parser.parse_args()

if args.verbose:
    print("Now start computing shap values...")

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

predictor_list = get_predictor_list(args.predictors)
if args.verbose:
    print("Number of hidden layers in the embeeding NN: {}".format(HIDDEN_LAYERS))
    print("Number of neurons in each NN layer: {}".format(NEURONS))
    print("Total number of training iterations: {}".format(TOTAL_ITER-1))
    print("Learning rate: {}".format(LEARNING_RATE))
    print("Number of timesteps: {}".format(NT))
    if args.normalize:
        print("Compute NORMALIZED SHAP values")
    else:
        print("Compute UNNORMALIZED SHAP values")
    print("Spatial predictors:")

    for p in predictor_list:
        print("+ {}".format(p))
    

create_folder_if_not_exists(shap_dir, verbose=args.verbose)


def predict_param(predictors, params,
                  model, var, dalec_par_dict,
                  pheno_par_dict, 
                  initial_par_dict, varmin, varmax):
    
    embedded_param_state, dalec_param_state, initial_param_state, pheno_param_state = params
    embedded_layer = embed_prediction_forward(embedded_param_state, predictors[:, 1:])
    dalec_parameters = model.unnormalize(parameter_prediction_forward(dalec_param_state, embedded_layer))
    if var in dalec_par_dict:
        return dalec_parameters[:, dalec_par_dict[var]] / (varmax - varmin)
    elif var in pheno_par_dict:
        augmented_pheno_predictors = jnp.concatenate([embedded_layer, predictors[:, 0:1]], axis=-1)
        pheno_parameters = model.unnormalize_pheno(parameter_prediction_forward(pheno_param_state, augmented_pheno_predictors))
        if var == "Bday" or var =="Fday":
            return pheno_parameters[:, pheno_par_dict[var]] % 365.25
        else:
            return pheno_parameters[:, pheno_par_dict[var]] / (varmax - varmin)
    elif var in initial_par_dict:
        initial_pools = model.unnormalize_pools(parameter_prediction_forward(initial_param_state, embedded_layer))
        return initial_pools[:, initial_par_dict[var]] / (varmax - varmin)
    elif var == "Q10_cal":
        return jnp.exp(10*dalec_parameters[:, dalec_par_dict["Q10"]]) / (varmax - varmin)
    elif var == "auto_resp_ratio":
        return dalec_parameters[:, dalec_par_dict["f_gpp"]] / (varmax - varmin)
    elif var == "leaf_allocation":
        return (1 - dalec_parameters[:, dalec_par_dict["f_gpp"]]) * dalec_parameters[:, dalec_par_dict["f_fol"]] / (varmax - varmin)
    elif var == "labile_allocation":
        return (1- dalec_parameters[:, dalec_par_dict["f_gpp"]] - (1 - dalec_parameters[:, dalec_par_dict["f_gpp"]]) * dalec_parameters[:, dalec_par_dict["f_fol"]]) * dalec_parameters[:, dalec_par_dict["flab"]] / (varmax - varmin)
    elif var == "root_allocation":
        return (1- dalec_parameters[:, dalec_par_dict["f_gpp"]]- (1 - dalec_parameters[:, dalec_par_dict["f_gpp"]]) * dalec_parameters[:, dalec_par_dict["f_fol"]] - (1- dalec_parameters[:, dalec_par_dict["f_gpp"]] - (1 - dalec_parameters[:, dalec_par_dict["f_gpp"]]) * dalec_parameters[:, dalec_par_dict["f_fol"]]) * dalec_parameters[:, dalec_par_dict["flab"]]) * dalec_parameters[:, dalec_par_dict["f_root"]]  / (varmax - varmin)
    elif var == "wood_allocation":
        return (1-dalec_parameters[:, dalec_par_dict["f_gpp"]]-((1 - dalec_parameters[:, dalec_par_dict["f_gpp"]]) * dalec_parameters[:, dalec_par_dict["f_fol"]]) - ((1- dalec_parameters[:, dalec_par_dict["f_gpp"]] - (1 - dalec_parameters[:, dalec_par_dict["f_gpp"]]) * dalec_parameters[:, dalec_par_dict["f_fol"]]) * dalec_parameters[:, dalec_par_dict["flab"]]) - ((1- dalec_parameters[:, dalec_par_dict["f_gpp"]]- (1 - dalec_parameters[:, dalec_par_dict["f_gpp"]]) * dalec_parameters[:, dalec_par_dict["f_fol"]] - (1- dalec_parameters[:, dalec_par_dict["f_gpp"]] - (1 - dalec_parameters[:, dalec_par_dict["f_gpp"]]) * dalec_parameters[:, dalec_par_dict["f_fol"]]) * dalec_parameters[:, dalec_par_dict["flab"]]) * dalec_parameters[:, dalec_par_dict["f_root"]])) / (varmax - varmin)


model = DALEC993(water_stress_type="default")


# read in spatial predictors values
predictor_matrix = read_multiple_varible_to_array(data_dir, "differland_global_driver_v6.nc", predictor_list)
# get the CMS-Flux index

RUN_SIMULATION_IDX = read_variable_to_vector(data_dir, "run_simulation_idx_v6.nc", "run_simulation_idx", time_idx=run-1)


VALID = read_variable_to_vector(data_dir, "era_valid_v6.nc", "era_valid")

INVALID = np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID) | (RUN_SIMULATION_IDX < 0) # filter out TEST PIXELS
predictor_matrix[:, INVALID] = np.nan
ASSIMILATE_BULK_FLAG = read_variable_to_vector(data_dir, "assimilate_bulk_variable_v6.nc", "assimilate_bulk_variable")

# filter out nan pixles
not_nan_idx = np.invert((np.sum(np.isnan(predictor_matrix), axis=0) > 0))
predictor_matrix = predictor_matrix[:, not_nan_idx]


VALID_IDX = RUN_SIMULATION_IDX[not_nan_idx]

shuffle_idx = np.argsort(VALID_IDX, kind='mergesort')
ASSIMILATE_SHUFFLE_FLAG = ASSIMILATE_BULK_FLAG[not_nan_idx][shuffle_idx]

sorted_valid_idx = VALID_IDX[shuffle_idx]
predictor_matrix_shuffled = predictor_matrix[:, shuffle_idx]

train_test_idx = np.round(np.max(sorted_valid_idx) * 0.9).astype(np.int32)

sorted_valid_idx_train = sorted_valid_idx[sorted_valid_idx <= train_test_idx]

predictor_matrix_train = predictor_matrix_shuffled[:, sorted_valid_idx <= train_test_idx]

scaled_predictor_matrix_train = (predictor_matrix_train.T - np.mean(predictor_matrix_train, axis=1)) / np.std(predictor_matrix_train, axis=1)

train_matrix = jnp.array(scaled_predictor_matrix_train, dtype=jnp.float32)

pickle_name = os.path.join(output_dir, exp_str + ".pickle")

if args.verbose:
    print("loading calibrated parameters from {}".format(pickle_name))
with open(pickle_name, "rb") as fp:
    param_state = pickle.load(fp)

if args.verbose:
    print("Parameters loaded.")
    print("Now calculating model performance...")

if args.normalize:
    param_ds = xr.open_dataset(os.path.join(nc_dir, "{}_params.nc".format(exp_str)))
    param_ds["Bday"] = param_ds.Bday % 365.25
    param_ds["Fday"] = param_ds.Fday % 365.25
    param_ds["auto_resp_ratio"] = param_ds.f_gpp
    param_ds["leaf_allocation"] = (1 - param_ds.f_gpp) * param_ds.f_fol
    param_ds["labile_allocation"] = (1- param_ds.f_gpp - (1 - param_ds.f_gpp) * param_ds.f_fol) * param_ds.flab
    param_ds["root_allocation"] = (1- param_ds.f_gpp- (1 - param_ds.f_gpp) * param_ds.f_fol - (1- param_ds.f_gpp - (1 - param_ds.f_gpp) * param_ds.f_fol) * param_ds.flab) * param_ds.f_root
    param_ds["wood_allocation"] = 1-param_ds["auto_resp_ratio"]-param_ds["leaf_allocation"]-param_ds["labile_allocation"]-param_ds["root_allocation"]
    param_ds["Q10_cal"]=np.exp(10*param_ds["Q10"])

    
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
"sif_beta_plus_three",
"p_fol",
"p_wood"
]

pheno_parnames = ["Bday", "Fday"]

initial_poolnames = ["Clab0", "Cfol0", "Croot0", "Cwood0", "Clitter0", "Csom0", "Water0"]

dalec_par_dict = {n:i for i, n in enumerate(dalec_parnames)}
pheno_par_dict = {n:i for i, n in enumerate(pheno_parnames)}
initial_par_dict = {n:i for i, n in enumerate(initial_poolnames)}

np.random.seed(run)

shap_sel_idx = np.random.randint(low=0, high=sorted_valid_idx_train.shape[0], size=1000)
shap_predictors = train_matrix[shap_sel_idx, :]
X = pd.DataFrame({predictor_list[i]:shap_predictors[:, i] for i in range(len(predictor_list))})
X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution


variables_list = ["clab_release_period", "p_wood", "sif_alpha",
            "LCMA", "Q10", "canopy_efficiency",
               "sif_beta_plus_three", "leaf_fall_period", "Bday", "Fday",
                "auto_resp_ratio", "leaf_allocation", "labile_allocation", "root_allocation",
                "wood_allocation", "Q10_cal"]

for var in variables_list:
    if args.verbose:
        print("Computing SHAP values for: {}".format(var))
    
    if args.normalize == "true":
        varmax = np.nanmax(param_ds[var].values)
        varmin = np.nanmin(param_ds[var].values)
    
        predict_fun = jax.jit(partial(predict_param, params=param_state, model=model, dalec_par_dict=dalec_par_dict,
                          pheno_par_dict=pheno_par_dict, 
                          initial_par_dict=initial_par_dict, var=var, varmin=varmin, varmax=varmax))        
    else:
        varmax = 2
        varmin = 1
    
        predict_fun = jax.jit(partial(predict_param, params=param_state, model=model, dalec_par_dict=dalec_par_dict,
                          pheno_par_dict=pheno_par_dict, 
                          initial_par_dict=initial_par_dict, var=var, varmin=varmin, varmax=varmax))
       
   
    explainer = shap.KernelExplainer(predict_fun, X100)
    shap_values = explainer(X)
    
    if args.normalize == "false":
        save_fn = os.path.join(shap_dir, "{}_{}_shap.pickle".format(exp_str, var))
    else:
        save_fn = os.path.join(shap_dir, "{}_{}_normalize_shap.pickle".format(exp_str, var))
        
    with open(save_fn, "wb") as f:
        pickle.dump(shap_values, f)

    if args.verbose:
        print("SHAP values saved to: {}".format(save_fn))

    