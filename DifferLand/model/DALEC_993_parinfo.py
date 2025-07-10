from collections import namedtuple
import jax.numpy as jnp

DALEC993ParamBounds = namedtuple("DALEC993ParamBounds", ["decomposition_rate",
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
                           "Bday",
                           "flab",
                           "clab_release_period",
                           "Fday",
                           "leaf_fall_period",
                           "LCMA",
                           "Clab",
                           "Cfol",
                           "Croot",
                           "Cwood",
                           "Clitter",
                           "Csom",
                           "IWUE",
                           "runoff_focal_point",
                           "field_capacity", 
                           "initial_water",
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
                           "sif_beta_plus_three", "p_fol", "p_wood"])

dalec993_parmin_arr = jnp.array([1.0e-5, 0.2e0, 0.01e0, 0.01e0, 1.001e0, 2.5e-5,
                                 0.0001e0, 0.0001e0, 1.0e-7, 0.018e0, 5.0e0, 365.25,
                                 0.01e0, 30.4375e0, 365.25, 30.4375e0, 5.0e0, 1.0e0,
                                 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 10.0e0, 1.0e0, 1.0e0,
                                 1.0e0, 0.01e0, 0.01e0, 0.01e0, 0.01e0, 1.001e0, 0.01e0, 0.5e0, 0.01e0, 0.01e0,
                                 3.0e0, 1.0e0, 0.0001, 0.000001], dtype=jnp.float32)

dalec993_parmin = DALEC993ParamBounds(*dalec993_parmin_arr)



dalec993_parmax_arr = jnp.array([0.01e0, 0.8e0, 0.5e0, 1.0e0, 8.0e0, 0.001e0, 0.01e0, 0.01e0,
                                 0.001e0, 0.08e0, 50.0e0, 365.25*4, 0.5e0, 100.0e0, 365.25*4,
                                 150.0e0, 200.0e0, 2000.0e0, 2000.0e0, 2000.0e0, 100000.0e0,
                                 2000.0e0, 200000.0e0, 50.0e0, 100000.0e0, 10000.0e0, 10000.0e0,
                                 1.0e0, 1.0e0, 1.0e0, 1.0e0, 8.0e0, 1.0e0, 30.0e0, 0.3e0, 0.5e0, 35.0e0, 9.0e0, 0.01, 0.00008],
                       dtype=jnp.float32)
          
          
dalec993_parmax = DALEC993ParamBounds(*dalec993_parmax_arr)

dalec993_param_parmax = dalec993_parmax_arr[jnp.array([1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,24,25,26,
                                                       28,29,30,31,32,33,34,35,36,37,38,39,40], dtype=jnp.int32)-1]

dalec993_param_parmin = dalec993_parmin_arr[jnp.array([1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,24,25,26,
                                                       28,29,30,31,32,33,34,35,36,37,38,39,40], dtype=jnp.int32)-1]


dalec993_pheno_parmin = dalec993_parmin_arr[jnp.array([12, 15], dtype=jnp.int32)-1]
dalec993_pheno_parmax = dalec993_parmax_arr[jnp.array([12, 15], dtype=jnp.int32)-1]


dalec993_pool_parmin = dalec993_parmin_arr[jnp.array([18,19,20,21,22,23,27], dtype=jnp.int32)-1]
dalec993_pool_parmax = dalec993_parmax_arr[jnp.array([18,19,20,21,22,23,27], dtype=jnp.int32)-1]

DALEC993Outputs = namedtuple("DALEC993Outputs", ["lai", "gpp", "ET", "temperate", "respiration_auto", "leaf_production", "labile_production", "root_production",
    "wood_production", "lff", "lrf", "labile_release", "leaf_litter", "wood_litter", "root_litter", "respiration_hetero_litter",
    "respiration_hetero_som", "litter_to_som", "runoff", "labile_fire_combust", "foliar_fire_combust", "root_fire_combust",
    "wood_fire_combust", "litter_fire_combust", "som_fire_combust", "labile_fire_transfer", "foliar_fire_transfer",
    "root_fire_transfer", "wood_fire_transfer", "litter_fire_transfer", "total_fire_combust", 
    "nee", "nbe", "next_labile_pool", "next_foliar_pool",
    "next_root_pool", "next_wood_pool", "next_litter_pool", "next_som_pool", "next_water_pool", "beta", "SIF", "violation", "vod"])


dalec993_pfn = DALEC993Outputs(*jnp.arange(44))

