import jax.numpy as jnp
from jax import Array
from typing import Callable, Union, Any


def sum_of_error(
    predicted: jnp.ndarray,
    true: jnp.ndarray,
    mask: jnp.ndarray,
    weight: Union[int, float] = 1
) -> jnp.ndarray:
    """
    Compute the weighted sum of squared errors between predicted and true values,
    considering only entries where the mask is nonzero.

    Parameters
    ----------
    predicted : jax.numpy.ndarray
        Predicted values.
    
    true : jax.numpy.ndarray
        True target values.
    
    mask : jax.numpy.ndarray
        Mask array of the same shape as `predicted` and `true`.
        Elements with mask value 0 are excluded from the error calculation.
    
    weight : int or float, optional
        Scalar multiplier applied to the final summed error (default is 1).

    Returns
    -------
    jax.numpy.ndarray
        Scalar tensor representing the weighted sum of squared errors.
    """
    return jnp.sum((predicted - true) ** 2 * mask) * weight


def patch_of_error(
    predicted: jnp.ndarray,
    true: jnp.ndarray,
    mask: jnp.ndarray,
    weight: Union[int, float] = 1
) -> jnp.ndarray:
    """
    Compute the weighted patch-level squared error between predicted and true values.

    This measures how the spatial mean of predictions deviates from the mean
    of true values in each feature (or spatial column), weighted by the average
    mask coverage.

    Parameters
    ----------
    predicted : jax.numpy.ndarray
        Predicted values, of shape (n_samples, n_features).
    
    true : jax.numpy.ndarray
        True target values, of the same shape as `predicted`.
    
    mask : jax.numpy.ndarray
        Mask array of the same shape as `predicted` and `true`.
        Elements with mask value 0 are excluded from the mean calculations.
    
    weight : int or float, optional
        Scalar multiplier applied to the final patch error (default is 1).

    Returns
    -------
    jax.numpy.ndarray
        Scalar tensor representing the weighted patch-level squared error.
    """
    return (
        jnp.sum(
            ((jnp.mean(predicted * mask, axis=0) - jnp.mean(true * mask, axis=0)) ** 2)
            * jnp.mean(mask, axis=0)
        )
        * weight
    )


def compute_edc_loss(
    result: jnp.ndarray,
    initial_pools_sample: jnp.ndarray,
    pfn: Any,
    epsilon: float = 1e-3
) -> jnp.ndarray:
    """
    Compute the Ecosystem Disequilibrium Cost (EDC) loss.

    This function calculates a squared log-ratio loss measuring how far
    simulated carbon pool sizes and fluxes deviate from initial conditions,
    accounting for internal ecosystem fluxes and average seasonal pools.

    It uses the ratio of mean annual influx to outflux for each carbon pool,
    adjusted by initial pool sizes and seasonal means. The resulting loss
    penalizes imbalances in the modeled carbon cycle and helps maintain
    mass balance constraints during model optimization.

    Parameters
    ----------
    result : jax.numpy.ndarray
        Array of shape (n_samples, n_timesteps, n_outputs) containing
        model outputs such as carbon fluxes and pools for each sample
        and timestep.
    
    initial_pools_sample : jax.numpy.ndarray
        Array of shape (n_samples, n_pools) with initial carbon pool
        sizes for each sample. The columns correspond to:
        labile, leaf, root, wood, litter, and SOM pools.
    
    pfn : Any
        Object or module providing integer indices into the third
        axis of `result` for various fluxes and pools. Must contain
        attributes such as:
        
        - labile_production
        - labile_release
        - labile_fire_combust
        - labile_fire_transfer
        - leaf_production
        - leaf_litter
        - foliar_fire_transfer
        - foliar_fire_combust
        - wood_production
        - wood_litter
        - wood_fire_combust
        - wood_fire_transfer
        - root_production
        - root_litter
        - root_fire_combust
        - root_fire_transfer
        - respiration_hetero_litter
        - litter_to_som
        - litter_fire_combust
        - litter_fire_transfer
        - respiration_hetero_som
        - som_fire_combust
        - next_labile_pool
        - next_leaf_pool
        - next_root_pool
        - next_wood_pool
        - next_litter_pool
        - next_som_pool

    epsilon : float, optional
        Small value to avoid division by zero or taking logarithm of zero
        (default is 1e-3).

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_samples,) containing the EDC loss for each sample.

    Notes
    -----
    - The loss penalizes deviations of modeled pools from initial values
      using squared log2 differences.
    - The January means are computed using timesteps 11, 23, 35, etc.,
      assuming monthly model outputs and January being timestep index 11
      in a zero-based indexing system.
    """
    labile_in = result[:, :, pfn.labile_production]
    labile_out = (
        result[:, :, pfn.labile_release]
        + result[:, :, pfn.labile_fire_combust]
        + result[:, :, pfn.labile_fire_transfer]
    )

    leaf_in = result[:, :, pfn.leaf_production] + result[:, :, pfn.labile_release]
    leaf_out = (
        result[:, :, pfn.leaf_litter]
        + result[:, :, pfn.foliar_fire_transfer]
        + result[:, :, pfn.foliar_fire_combust]
    )

    wood_in = result[:, :, pfn.wood_production]
    wood_out = (
        result[:, :, pfn.wood_litter]
        + result[:, :, pfn.wood_fire_combust]
        + result[:, :, pfn.wood_fire_transfer]
    )

    root_in = result[:, :, pfn.root_production]
    root_out = (
        result[:, :, pfn.root_litter]
        + result[:, :, pfn.root_fire_combust]
        + result[:, :, pfn.root_fire_transfer]
    )

    litter_in = (
        result[:, :, pfn.leaf_litter]
        + result[:, :, pfn.root_litter]
        + result[:, :, pfn.labile_fire_transfer]
        + result[:, :, pfn.foliar_fire_transfer]
        + result[:, :, pfn.root_fire_transfer]
    )
    litter_out = (
        result[:, :, pfn.respiration_hetero_litter]
        + result[:, :, pfn.litter_to_som]
        + result[:, :, pfn.litter_fire_combust]
        + result[:, :, pfn.litter_fire_transfer]
    )

    som_in = (
        result[:, :, pfn.wood_litter]
        + result[:, :, pfn.litter_to_som]
        + result[:, :, pfn.wood_fire_transfer]
        + result[:, :, pfn.litter_fire_transfer]
    )
    som_out = (
        result[:, :, pfn.respiration_hetero_som]
        + result[:, :, pfn.som_fire_combust]
    )

    mean_labile_ratio = jnp.sum(labile_in, axis=1) / jnp.maximum(
        jnp.sum(labile_out, axis=1), epsilon
    )
    mean_leaf_ratio = jnp.sum(leaf_in, axis=1) / jnp.maximum(
        jnp.sum(leaf_out, axis=1), epsilon
    )
    mean_root_ratio = jnp.sum(root_in, axis=1) / jnp.maximum(
        jnp.sum(root_out, axis=1), epsilon
    )
    mean_wood_ratio = jnp.sum(wood_in, axis=1) / jnp.maximum(
        jnp.sum(wood_out, axis=1), epsilon
    )
    mean_litter_ratio = jnp.sum(litter_in, axis=1) / jnp.maximum(
        jnp.sum(litter_out, axis=1), epsilon
    )
    mean_som_ratio = jnp.sum(som_in, axis=1) / jnp.maximum(
        jnp.sum(som_out, axis=1), epsilon
    )

    initial_labile = initial_pools_sample[:, 0]
    initial_leaf = initial_pools_sample[:, 1]
    initial_root = initial_pools_sample[:, 2]
    initial_wood = initial_pools_sample[:, 3]
    initial_litter = initial_pools_sample[:, 4]
    initial_som = initial_pools_sample[:, 5]

    mean_labile_jan = jnp.mean(result[:, 11::12, pfn.next_labile_pool], axis=1)
    mean_leaf_jan = jnp.mean(result[:, 11::12, pfn.next_leaf_pool], axis=1)
    mean_wood_jan = jnp.mean(result[:, 11::12, pfn.next_wood_pool], axis=1)
    mean_root_jan = jnp.mean(result[:, 11::12, pfn.next_root_pool], axis=1)
    mean_litter_jan = jnp.mean(result[:, 11::12, pfn.next_litter_pool], axis=1)
    mean_som_jan = jnp.mean(result[:, 11::12, pfn.next_som_pool], axis=1)

    initial_labile_ratio = mean_labile_ratio * mean_labile_jan / initial_labile
    initial_leaf_ratio = mean_leaf_ratio * mean_leaf_jan / initial_leaf
    initial_root_ratio = mean_root_ratio * mean_root_jan / initial_root
    initial_wood_ratio = mean_wood_ratio * mean_wood_jan / initial_wood
    initial_litter_ratio = mean_litter_ratio * mean_litter_jan / initial_litter
    initial_som_ratio = mean_som_ratio * mean_som_jan / initial_som

    labile_edc = (jnp.abs(jnp.log(initial_labile_ratio + epsilon)) / jnp.log(2)) ** 2
    leaf_edc = (jnp.abs(jnp.log(initial_leaf_ratio + epsilon)) / jnp.log(2)) ** 2
    root_edc = (jnp.abs(jnp.log(initial_root_ratio + epsilon)) / jnp.log(2)) ** 2
    wood_edc = (jnp.abs(jnp.log(initial_wood_ratio + epsilon)) / jnp.log(2)) ** 2
    litter_edc = (jnp.abs(jnp.log(initial_litter_ratio + epsilon)) / jnp.log(2)) ** 2
    som_edc = (jnp.abs(jnp.log(initial_som_ratio + epsilon)) / jnp.log(2)) ** 2

    edc_loss = labile_edc + leaf_edc + root_edc + wood_edc + litter_edc + som_edc

    return edc_loss


def loss_fn_with_edc(
    params: Array,
    predictors: Array,
    met: Array,
    labels: Array,
    batch_forward: Callable[[Array, Array, Array], tuple[Array, Array]],
    pfn: Any,
    warm_up: int = 0,
    batch_size: int = 320,
    epsilon: float = 0.001,
) -> Union[float, Array]:
    """
    Compute the total loss for the model, including Ecological and Dynamical Constraints (EDC).

    This loss function evaluates how well the model reproduces various observed
    variables (fluxes, stocks, and other ecological quantities) and applies
    penalties for violations of ecological and dynamical constraints.

    The total loss includes the following components:
    - SIF loss (Solar-Induced Fluorescence)
    - LAI loss (Leaf Area Index)
    - Biomass loss (annual)
    - NBE loss (Net Biome Exchange)
    - NBE interannual variability loss
    - Fire loss
    - SOM loss (Soil Organic Matter)
    - Water pool loss
    - VOD loss (Vegetation Optical Depth, annual)
    - FluxNet GPP, RECO, and ET losses
    - GLEAM ET loss
    - Violation loss (sum of violations predicted by the model)
    - EDC loss

    Parameters
    ----------
    params : Array
        Model parameters to be evaluated. These are typically the normalized
        parameters being optimized.

    predictors : Array
        Predictor variables used by the model, e.g. climate drivers, site variables.

    met : Array
        Meteorological forcing data for the model.

    labels : Array
        Observational data to compare model outputs against. This includes:
        - fluxes
        - stocks
        - masks and relative weights indicating valid observations

    batch_forward : Callable[[Array, Array, Array], tuple[Array, Array]]
        Function that performs a forward simulation of the model:
            initial_pools, result = batch_forward(params, predictors, met)

        - `initial_pools` : Array of model state pools at start.
        - `result` : Array of model outputs with shape (batch_size, time, variables).

    pfn : Any
        Object holding index mappings into `result` for different output variables.
        E.g. pfn.SIF, pfn.lai, etc.

    warm_up : int, optional
        Number of initial timesteps to skip from loss computation (default is 0).
        Used to avoid initial model spin-up effects.

    batch_size : int, optional
        Number of samples in the batch. Default is 320.

    epsilon : float, optional
        Small constant to avoid division by zero in normalization operations.

    Returns
    -------
    total_loss : float or Array
        Total computed loss value, combining all error components and EDC penalties.

    Notes
    -----
    - The function aggregates multiple error terms using different weights.
    - EDC (Ecological and Dynamical Constraints) helps enforce ecological realism
      and system stability in the model by penalizing physically or biologically
      implausible states or parameter combinations.
    """

    initial_pools, result = batch_forward(params, predictors, met)
    result = result.squeeze()

    sif_loss = sum_of_error(
        result[:, warm_up:, pfn.SIF],
        labels[:, warm_up:, 0],
        labels[:, warm_up:, 1],
        weight=400,
    )

    lai_loss = sum_of_error(
        result[:, warm_up:, pfn.lai],
        labels[:, warm_up:, 4],
        labels[:, warm_up:, 5],
        weight=10,
    )

    modeled_biomass = (
        result[:, warm_up:, pfn.next_labile_pool]
        + result[:, warm_up:, pfn.next_foliar_pool]
        + result[:, warm_up:, pfn.next_wood_pool]
        + result[:, warm_up:, pfn.next_root_pool]
    )
    observed_biomass = labels[:, warm_up:, 8]
    biomass_mask = labels[:, warm_up:, 9]

    modeled_biomass_annual = jnp.mean(
        modeled_biomass.reshape(batch_size, -1, 12), axis=2
    )
    observed_biomass_annual = jnp.mean(
        observed_biomass.reshape(batch_size, -1, 12), axis=2
    )
    biomass_mask_annual = jnp.prod(
        biomass_mask.reshape(biomass_mask.shape[0], -1, 12), axis=2
    )

    biomass_loss = sum_of_error(
        modeled_biomass_annual,
        observed_biomass_annual,
        biomass_mask_annual,
        weight=5e-5,
    )

    modeled_som_clim = jnp.mean(result[:, warm_up:, pfn.next_som_pool], axis=1)
    som_loss = sum_of_error(
        modeled_som_clim,
        labels[:, 0, 12],
        labels[:, 0, 13],
        weight=3e-6
    )

    nbe_loss = patch_of_error(
        result[:, warm_up:, pfn.nbe],
        labels[:, warm_up:, 2],
        labels[:, warm_up:, 3],
        weight=3000,
    )

    modeled_nbe_annual = jnp.mean(
        result[:, warm_up:, pfn.nbe].reshape(batch_size, -1, 12), axis=2
    )
    observed_nbe_annual = jnp.mean(
        labels[:, warm_up:, 2].reshape(batch_size, -1, 12), axis=2
    )
    nbe_mask_annual = jnp.prod(
        labels[:, warm_up:, 3].reshape(batch_size, -1, 12), axis=2
    )

    nbe_iav_loss = patch_of_error(
        modeled_nbe_annual,
        observed_nbe_annual,
        nbe_mask_annual,
        weight=3000 * 3.46,
    )

    fire_loss = patch_of_error(
        result[:, warm_up:, pfn.total_fire_combust],
        labels[:, warm_up:, 10],
        labels[:, warm_up:, 11],
        weight=1000,
    )

    violation_loss = jnp.sum(result[:, :, pfn.violation]) * 400

    predicted_water_pool_mean = jnp.sum(
        result[:, :, pfn.next_water_pool] * labels[:, :, 7], axis=1
    ) / (jnp.sum(labels[:, :, 7], axis=1) + epsilon)

    water_loss = patch_of_error(
        ((result[:, warm_up:, pfn.next_water_pool]).T - predicted_water_pool_mean).T,
        labels[:, warm_up:, 6],
        labels[:, warm_up:, 7],
        weight=0.2,
    )

    modeled_vod = result[:, warm_up:, pfn.vod]
    observed_vod = labels[:, warm_up:, 14]
    vod_mask = labels[:, warm_up:, 15]

    modeled_vod_annual = jnp.mean(modeled_vod.reshape(batch_size, -1, 12), axis=2)
    observed_vod_annual = jnp.mean(observed_vod.reshape(batch_size, -1, 12), axis=2)
    vod_mask_annual = jnp.prod(vod_mask.reshape(vod_mask.shape[0], -1, 12), axis=2)

    vod_loss = sum_of_error(
        modeled_vod_annual,
        observed_vod_annual,
        vod_mask_annual,
        weight=100,
    )

    fluxnet_gpp_loss = sum_of_error(
        result[:, warm_up:, pfn.gpp],
        labels[:, warm_up:, 16],
        labels[:, warm_up:, 17],
        weight=100,
    )

    fluxnet_reco_loss = sum_of_error(
        result[:, warm_up:, pfn.gpp] + result[:, warm_up:, pfn.nee],
        labels[:, warm_up:, 18],
        labels[:, warm_up:, 19],
        weight=100,
    )

    fluxnet_et_loss = sum_of_error(
        result[:, warm_up:, pfn.ET],
        labels[:, warm_up:, 20],
        labels[:, warm_up:, 21],
        weight=100,
    )

    gleam_et_loss = sum_of_error(
        result[:, warm_up:, pfn.ET],
        labels[:, warm_up:, 22],
        labels[:, warm_up:, 23],
        weight=1,
    )

    edc_loss = (
        jnp.sum(
            compute_edc_loss(result, initial_pools, pfn, epsilon=epsilon)
            * labels[:, 0, 5]
        )
        * 10
    )

    return (
        sif_loss
        + lai_loss
        + biomass_loss
        + nbe_loss
        + violation_loss
        + fire_loss
        + som_loss
        + water_loss
        + vod_loss
        + fluxnet_gpp_loss
        + fluxnet_reco_loss
        + fluxnet_et_loss
        + gleam_et_loss
        + nbe_iav_loss
        + edc_loss
    )

