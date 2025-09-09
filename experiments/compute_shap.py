import argparse
import jax
import jax.numpy as jnp
from functools import partial
import os
import numpy as np
import pandas as pd
import pickle

import sys

sys.path.insert(1, "..")
from DifferLand.util.preprocessing import (
    read_variable_to_vector,
    read_multiple_varible_to_array,
)
from DifferLand.optimization.forward import (
    embed_prediction_forward,
    parameter_prediction_forward,
)

from DifferLand.model.DALEC993 import DALEC993
import shap
from DifferLand.util.preprocessing import create_folder_if_not_exists

def parse_int_list(s):
    """Parse a comma-delimited string of integers into a list."""
    try:
        return [int(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer list: '{s}'")
    
parser = argparse.ArgumentParser(
    prog="Compute SHAP values for the spatilization network.",
    description="JAX implementation of differentiable CARDAMOM!",
    epilog="By Jianing Fang",
)

parser.add_argument(
    "-r",
    "--runs",
    type=parse_int_list,
    help="Specify an ensemble member index or a comma-separated list" \
    "of indices for SHAP computation (e.g., 1 or 1,2,3).",
    required=True,
)
parser.add_argument("-p", "--predictors", required=True)
parser.add_argument("-t", "--target", type=str, required=True)
parser.add_argument(
    "-v", "--verbose", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "-e", "--ensemble_shap", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "-c", "--pft", type=str, default="none"
)
parser.add_argument(
    "-x", "--pft_purity_percentage_threshold", type=float, default=80.0
)
parser.add_argument(
    "-n", "--normalize", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "-d", "--combined_training_and_development_sets",
    help="whether the model was train on combined training and develoment sets",
    action=argparse.BooleanOptionalAction, default=False
)

def parse_int_list(s):
    """Parse a comma-delimited string of integers into a list."""
    try:
        return [int(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer list: '{s}'")
    
OUTPUT_DIR = "./output/"
LOG_DIR = "./log/"
POSTANALYSIS_DIR = "./postanalysis/"

NC_DIR = "./postanalysis/nc/"
SHAP_DIR = "./postanalysis/shap/"
DATA_DIR = "/burg-archive/glab/users/jf3423/data/CARDAMOM_driver_data/global/"
predictor_mean_df = pd.read_csv(os.path.join(DATA_DIR, "predictor_mean_df_v6.csv"))
predictor_std_df = pd.read_csv(os.path.join(DATA_DIR, "predictor_std_df_v6.csv")) 
   
    
create_folder_if_not_exists(POSTANALYSIS_DIR)
create_folder_if_not_exists(SHAP_DIR)

args = parser.parse_args()

if args.pft not in ["NF", "DBF", "EBF", "MF", "SH", "SAV", "GRA", "WET", "CRO", "NVG"]:
    raise ValueError("--pft must be one of `none`, `NF`, `DBF`, `EBF`, `MF`," /
                     "`SH`, `SAV`, `GRA`, `WET`, `CRO`, `NVG`, got {}".format(args.pft))


        
    

model = DALEC993(water_stress_type="default")
VARIABLE_OF_INTEREST = args.target
RUNS = np.array(args.runs)
SETUP = args.predictors


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

def param_prediction_forward(predictors,
                             model,
                             var,
                             varmin,
                             varmax,
                             dalec_par_dict,
                             pheno_par_dict,
                             initial_par_dict,
                             params,
                             mean,
                             std
    ):
    normalized_predictors = (predictors - mean) / std
    embedded_param_state, dalec_param_state, initial_param_state, pheno_param_state = params
    embedded_layer = embed_prediction_forward(embedded_param_state, normalized_predictors[:, 1:])
    dalec_parameters = model.unnormalize(parameter_prediction_forward(dalec_param_state, embedded_layer))
    
   
    if var in dalec_par_dict:
        return dalec_parameters[:, dalec_par_dict[var]]
    elif var in pheno_par_dict:
        augmented_pheno_predictors = jnp.concatenate([embedded_layer, predictors[:, 0:1]], axis=-1)
        pheno_parameters = model.unnormalize_pheno(parameter_prediction_forward(pheno_param_state, 
                                                                                augmented_pheno_predictors))
        if var == "Bday" or var =="Fday":
            return pheno_parameters[:, pheno_par_dict[var]] % 365.25
        else:
            return pheno_parameters[:, pheno_par_dict[var]] / (varmax - varmin)
    elif var in initial_par_dict:
        initial_pools = model.unnormalize_pools(parameter_prediction_forward(initial_param_state, embedded_layer))
        return initial_pools[:, initial_par_dict[var]] / (varmax - varmin)
    else:
        f_gpp = dalec_parameters[:, dalec_par_dict["f_gpp"]]
        f_fol = dalec_parameters[:, dalec_par_dict["f_fol"]]
        flab = dalec_parameters[:, dalec_par_dict["f_fol"]]
        f_root = dalec_parameters[:, dalec_par_dict["f_root"]]
        auto_resp_ratio = f_gpp
        leaf_allocation = (1 - f_gpp) * f_fol
        labile_allocation = (1- f_gpp - (1 - f_gpp) * f_fol) * flab
        root_allocation = (1 - f_gpp- (1 - f_gpp) * f_fol - (1- f_gpp - (1 - f_gpp) * f_fol) * flab) * f_root
        wood_allocation = 1- auto_resp_ratio - leaf_allocation - labile_allocation - root_allocation
        Q10=jnp.exp(10*dalec_parameters[:, dalec_par_dict["Q10"]])
        if var == "Q10_cal":
            return Q10 / (varmax - varmin)
        if var == "auto_resp_ratio":
            return auto_resp_ratio / (varmax - varmin)
        if var == "leaf_allocation":
            return leaf_allocation / (varmax - varmin)
        if var == "labile_allocation":
            return labile_allocation / (varmax - varmin)
        if var == "root_allocation":
            return root_allocation / (varmax - varmin)
        if var == "wood_allocation":
            return wood_allocation / (varmax - varmin)    
        raise ValueError("Target name {} is not supported.".format(VARIABLE_OF_INTEREST))

ensemble_param_prediction_forward = jax.vmap(partial(param_prediction_forward),
                                             in_axes=[None, None, None, None, 
                                                      None, None, None, None,
                                                      0, 0, 0]
                                            )


def predict_batch_averaged_param(predictors,
                 ensemble_predict_func,
                 model,
                 var,
                 varmin,
                 varmax,
                 dalec_par_dict,
                 pheno_par_dict,
                 initial_par_dict,
                 batch_params,
                 batch_mean,
                 batch_std
):
    predicted = ensemble_predict_func(predictors,
                                      model,
                                      var,
                                      varmin,
                                      varmax,
                                      dalec_par_dict,
                                      pheno_par_dict,
                                      initial_par_dict,
                                      batch_params,
                                      batch_mean,
                                      batch_std)
    return jnp.mean(predicted, axis=0)

def get_predictor_list(setup):
    predictor_list = ["LAT"]
    if "PFT" in setup:
        predictor_list += ["NF", "DBF", "EBF", "MF", "SH", "SAV",
                           "GRA", "WET", "CRO", "NVG"]
    if "CLIM" in setup:
        predictor_list += ["MAT", "MAP", "elevation"]
    if "AGE" in setup:
        predictor_list += ["canopy_height", "tree_age_000_filled"]
    if "SOIL" in setup:
        predictor_list += ["BULK_DEN", "SAND", "SILT", "CLAY", "GRAVEL"]
    if "LATLON" in setup:
        predictor_list += ["lat_deg", "lon_deg"]
    if "CONTROL" in setup:
        predictor_list += ["null"]
    return predictor_list


predictor_list = get_predictor_list(SETUP)
param_state_list = []
for run in RUNS:
    exp_str = "dalec993_{}_run_{}".format(args.predictors, run)
    pickle_name = os.path.join(OUTPUT_DIR, exp_str + ".pickle")
    with open(pickle_name, "rb") as fp:
        param_state = pickle.load(fp)
        param_state_list.append(param_state)
        
# read in spatial predictors values
predictor_matrix = read_multiple_varible_to_array(DATA_DIR, "combined_global_initial_v6.nc", predictor_list)


RUN_SIMULATION_IDX = read_variable_to_vector(DATA_DIR, "run_simulation_idx_v6.nc", "run_simulation_idx", time_idx=60)
    
VALID = read_variable_to_vector(DATA_DIR, "era_valid_v6.nc", "era_valid")

INVALID = np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID) | (RUN_SIMULATION_IDX < 0) # filter out TEST PIXELS

ASSIMILATE_BULK_FLAG = read_variable_to_vector(DATA_DIR, "assimilate_bulk_variable_v6.nc", "assimilate_bulk_variable")
predictor_matrix[:, INVALID] = np.nan

# filter out nan pixles
not_nan_idx = np.invert((np.sum(np.isnan(predictor_matrix), axis=0) > 0))
predictor_matrix = predictor_matrix[:, not_nan_idx]


VALID_IDX = RUN_SIMULATION_IDX[not_nan_idx]

shuffle_idx = np.argsort(VALID_IDX, kind='mergesort')
ASSIMILATE_SHUFFLE_FLAG = ASSIMILATE_BULK_FLAG[not_nan_idx][shuffle_idx]

sorted_valid_idx = VALID_IDX[shuffle_idx]
predictor_matrix_shuffled = predictor_matrix[:, shuffle_idx]

if args.pft != "none":
    igbp_mask = read_variable_to_vector(DATA_DIR,  "combined_global_initial_v6.nc", args.pft) > args.pft_purity_percentage_threshold
    igbp_mask = igbp_mask[shuffle_idx]
    predictor_matrix_shuffled = predictor_matrix_shuffled[:, igbp_mask]

        
subsampled_predictors = predictor_matrix_shuffled[:, np.random.randint(0, 
                                            predictor_matrix_shuffled.shape[1], size=1000)]
    
batch_params = jax.tree_util.tree_map(
    lambda *xs: jnp.stack(xs), *param_state_list
)


if not args.combined_training_and_development_sets:

    batch_mean = predictor_mean_df[predictor_list].loc[RUNS-1, :].values

    batch_std = predictor_std_df[predictor_list].loc[RUNS-1, :].values
else: 
    batch_mean = np.repeat(predictor_mean_df[predictor_list].loc[100, :].values[np.newaxis, :], len(RUNS), axis=0)
    batch_std = np.repeat(predictor_std_df[predictor_list].loc[100, :].values[np.newaxis, :], len(RUNS), axis=0)
    


X = pd.DataFrame({predictor_list[i]:subsampled_predictors[i, :] for i in range(len(predictor_list))})

if args.verbose:
    if args.pft == "none":
        print(f"Now start computing unconditioned SHAP values for {VARIABLE_OF_INTEREST} from {exp_str}...")
    else:
        print(f"Now start computing SHAP values for {VARIABLE_OF_INTEREST} from {exp_str} conditioned on PFT={args.pft}...")
        
X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution

predict_fun = jax.jit(partial(predict_batch_averaged_param,
                            ensemble_predict_func=ensemble_param_prediction_forward,
                            model=model,
                            var=VARIABLE_OF_INTEREST,
                            varmax=2,
                            varmin=1,
                            dalec_par_dict=dalec_par_dict,
                            pheno_par_dict=pheno_par_dict,
                            initial_par_dict=initial_par_dict,
                            batch_params=batch_params,
                            batch_mean=batch_mean,
                            batch_std=batch_std,
                            ))

explainer = shap.KernelExplainer(predict_fun, X100)
shap_values = explainer(X)


if args.pft =="none":  
    save_fn = os.path.join("./postanalysis/shap/", f"{SETUP}_{VARIABLE_OF_INTEREST}_{"-".join([str(r) for r in RUNS])}_ensemble_shap.py")
else:
    save_fn = os.path.join("./postanalysis/shap/", f"{SETUP}_{VARIABLE_OF_INTEREST}_{args.pft}_{"-".join([str(r) for r in RUNS])}_ensemble_shap.py")
    
with open(save_fn, "wb") as f:
    pickle.dump(shap_values, f)