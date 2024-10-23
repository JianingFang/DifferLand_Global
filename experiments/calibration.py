import argparse
# construct command line parser
parser = argparse.ArgumentParser(
                    prog='HCM',
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


# parse command line arguments
args = parser.parse_args()
if args.verbose:
    print("Welcome to DifferLand!")

HIDDEN_LAYERS = args.hidden_layers
NEURONS = args.neurons
TOTAL_ITER = args.iterations + 1
LEARNING_RATE = args.learning_rate
NT = args.number_of_timesteps

if args.run < 1 or args.run > 100:
    print("Error: RUN_IDX must be an integer between 1 and 100 inclusive.")
    exit()
run = args.run

# import external pacakges
import jax
import jax.numpy as jnp
from functools import partial
import optax
import os
import numpy as np
import logging
import pickle
import argparse
import sys
from tqdm import tqdm
# import the DifferLand modules
sys.path.insert(1, '../')
from DifferLand.util.preprocessing import read_variable_to_vector, read_multiple_varible_to_array
from DifferLand.util.preprocessing import read_multiple_variable_temporal_to_vector
from DifferLand.util.preprocessing import nan_read_multiple_variable_temporal_to_vector
from DifferLand.util.preprocessing import generate_data_loader, generate_input_loader
from DifferLand.util.preprocessing import create_folder_if_not_exists
from DifferLand.optimization.forward import embed_prediction_forward, parameter_prediction_forward
from DifferLand.optimization.loss_functions import *
from DifferLand.util.init_mlp_params import init_mlp_params
from DifferLand.model.DALEC993 import DALEC993


# define directories for accessing data and storing outputs
data_dir = "../data/"
output_dir = "./output/"
log_dir = "./log/"
create_folder_if_not_exists(output_dir, verbose=args.verbose)
create_folder_if_not_exists(log_dir, verbose=args.verbose)



def forward(params, predictors, met, model):
    embedded_param_state, dalec_param_state, initial_param_state, pheno_param_state = params
    embedded_layer = embed_prediction_forward(embedded_param_state, predictors[1:])
    dalec_parameters = model.unnormalize(parameter_prediction_forward(dalec_param_state, embedded_layer))
    augmented_pheno_predictors = jnp.concatenate([embedded_layer, predictors[0:1]])
    pheno_parameters = model.unnormalize_pheno(parameter_prediction_forward(pheno_param_state, augmented_pheno_predictors))
    initial_pools = model.unnormalize_pools(parameter_prediction_forward(initial_param_state, embedded_layer))
    final_state, all_fluxes = jax.lax.scan(partial(model.step, gpp_params=None, pheno_parameters=pheno_parameters,
                                                   dalec_parameters=dalec_parameters), initial_pools, met)
    return all_fluxes

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

predictor_list = get_predictor_list(args.predictors)
if args.verbose:
    print("Number of hidden layers in the embeeding NN: {}".format(HIDDEN_LAYERS))
    print("Number of neurons in each NN layer: {}".format(NEURONS))
    print("Total number of training iterations: {}".format(TOTAL_ITER-1))
    print("Learning rate: {}".format(LEARNING_RATE))
    print("Number of timesteps: {}".format(NT))
    print("Spatial predictors:")
    for p in predictor_list:
        print("+ {}".format(p))

exp_str = "dalec993_{}_run_{}".format(args.predictors,
                                run)

model = DALEC993(water_stress_type="default")

forward = partial(forward, model=model)
batch_forward = jax.jit(jax.vmap(jax.jit(forward), in_axes=[None, 0, 0]))

if args.verbose:
    print("Reading in datasets for model training...")
# read in spatial predictors
predictor_matrix = read_multiple_varible_to_array(data_dir, "differland_global_driver_v5.4.nc", predictor_list)
# get the CMS-Flux index
RUN_SIMULATION_IDX = read_variable_to_vector(data_dir, "run_simulation_idx_v5.4.nc", "run_simulation_idx", time_idx=run-1)

VALID = read_variable_to_vector(data_dir, "era_valid_v5.4.nc", "era_valid")

INVALID = np.isnan(RUN_SIMULATION_IDX) | np.invert(VALID) | (RUN_SIMULATION_IDX < 0) # filter out TEST PIXELS
predictor_matrix[:, INVALID] = np.nan
ASSIMILATE_BULK_FLAG = read_variable_to_vector(data_dir, "assimilate_bulk_variable_v5.4.nc", "assimilate_bulk_variable")

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
sorted_valid_idx_test = sorted_valid_idx[sorted_valid_idx > train_test_idx]

predictor_matrix_train = predictor_matrix_shuffled[:, sorted_valid_idx <= train_test_idx]
predictor_matrix_test = predictor_matrix_shuffled[:, sorted_valid_idx > train_test_idx]

scaled_predictor_matrix_train = (predictor_matrix_train.T - np.mean(predictor_matrix_train, axis=1)) / np.std(predictor_matrix_train, axis=1)
scaled_predictor_matrix_test = (predictor_matrix_test.T - np.mean(predictor_matrix_train, axis=1)) / np.std(predictor_matrix_train, axis=1)


train_matrix = jnp.array(scaled_predictor_matrix_train, dtype=jnp.float32)
test_matrix = jnp.array(scaled_predictor_matrix_test, dtype=jnp.float32)

met_list = ["DAYS", "T_min", "T_max", "SOLR", "CO2", "DOY", "BURNED_AREA", "VPD", "PREC", "LAT", "DELTA_T", "MAT", "MAP"]
met_matrix = read_multiple_variable_temporal_to_vector(data_dir, "differland_global_driver_v5.4.nc", met_list, not_nan_idx, shuffle_idx, n_t=NT)
met_matrix = jnp.transpose(met_matrix, axes=[2, 1, 0])
met_matrix_train = jnp.array(met_matrix[sorted_valid_idx <= train_test_idx, :], dtype=jnp.float32)
met_matrix_test = jnp.array(met_matrix[sorted_valid_idx > train_test_idx, :], dtype=jnp.float32)

output_list = ["SIF", "NBE", "LAI", "LWE_normalized", "agb_yan", "fire_emission", "som_const", "VOD", "gpp_fluxnet", "reco_fluxnet", "et_fluxnet", "ET"]
output_matrix = nan_read_multiple_variable_temporal_to_vector(data_dir, "differland_global_driver_v5.4.nc", output_list, not_nan_idx, shuffle_idx, n_t=NT)
output_matrix = jnp.transpose(output_matrix, axes=[2, 1, 0])

output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 2].set(-9999)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 3].set(0)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 6].set(-9999)
output_matrix=output_matrix.at[np.invert(ASSIMILATE_SHUFFLE_FLAG), :, 7].set(0)
output_matrix_train = jnp.array(output_matrix[sorted_valid_idx <= train_test_idx, :], dtype=jnp.float32)
output_matrix_test = jnp.array(output_matrix[sorted_valid_idx > train_test_idx, :], dtype=jnp.float32)

train_MET = generate_data_loader(met_matrix_train, sorted_valid_idx_train, zero_padding=False)
test_MET = generate_data_loader(met_matrix_test, sorted_valid_idx_test, zero_padding=False)

train_Y = generate_data_loader(output_matrix_train, sorted_valid_idx_train, zero_padding=True)
test_Y = generate_data_loader(output_matrix_test, sorted_valid_idx_test, zero_padding=True)

train_X = generate_input_loader(train_matrix, sorted_valid_idx_train)
test_X = generate_input_loader(test_matrix, sorted_valid_idx_test)

if args.verbose:
    print("Data for model calibration have been successfully loaded.")

if args.verbose:
    print("Initializing model parameters...")

embedded_param_state = init_mlp_params([len(predictor_list)-1] + [NEURONS,] * HIDDEN_LAYERS, n=np.random.randint(99999999))

dalec_param_state = init_mlp_params([NEURONS, len(model.param_parmin)], n=np.random.randint(99999999))

initial_param_state = init_mlp_params([NEURONS, len(model.pool_parmin)], n=np.random.randint(99999999))

pheno_param_state = init_mlp_params([NEURONS+1, len(model.pheno_parmin)], n=np.random.randint(99999999))

param_state = (embedded_param_state, dalec_param_state, initial_param_state, pheno_param_state)

loss_fn = partial(loss_993_mse_with_fluxnet, batch_forward=batch_forward, pfn=model.pfn)

                        
loss_grad_fn = jax.jit(jax.value_and_grad(jax.jit(loss_fn)))
tx = optax.adam(learning_rate=LEARNING_RATE)
opt_state = tx.init(param_state)

@jax.jit
def update(params, predictors, met, labels, opt_state):
    loss, grads = loss_grad_fn(params, predictors, met, labels)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

logging.basicConfig(level=logging.INFO,
                    filename="./{}/{}.log".format(log_dir, exp_str),
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
if args.verbose:
    print("Training start!")

for j in tqdm(range(1, TOTAL_ITER)):
    batch_loss = 0
    for (PREDICTORS, MET, LABELS) in zip(train_X, train_MET, train_Y):
        loss, param_state, opt_state = update(param_state, PREDICTORS, MET, LABELS, opt_state)
        batch_loss += loss
    logging.info("batch {}, loss: {:.2f}".format(j, batch_loss))
    if j % 10 == 0:
        test_loss = 0
        for (PREDICTORS, MET, LABELS) in zip(test_X, test_MET, test_Y):
            test_loss_batch = loss_fn(param_state, PREDICTORS, MET, LABELS)
            test_loss += test_loss_batch
        logging.info("VALIDATION LOSS {:.2f}".format(test_loss)) 

with open(os.path.join(output_dir, exp_str+".pickle"), "wb") as fp:
    pickle.dump(param_state, fp)

if args.verbose:
    print("Training complete!")
    print("Calibrated parameters saved to: {}".format(os.path.join("./{}/".format(output_dir), exp_str+".pickle")))