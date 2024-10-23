import jax.numpy as jnp

def sum_of_error(predicted, true, mask, weight=1):
    return jnp.sum((predicted-true)**2 * mask) * weight

def patch_of_error(predicted, true, mask, weight=1):
    return jnp.sum(((jnp.mean(predicted * mask, axis=0) - jnp.mean(true * mask, axis=0))**2) * jnp.mean(mask, axis=0)) * weight

    
def loss_996_mse_with_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=500)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=50)
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    predicted_water_pool_mean = jnp.sum(result[:, warm_up:, pfn.next_water_pool] * labels[:, warm_up:, 7], axis=1) / (jnp.sum(labels[:, warm_up:, 7], axis=1) + 0.01)
    water_loss = patch_of_error((result[:, warm_up:, pfn.next_water_pool].T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)

    return sif_loss+lai_loss+agb_loss+nbe_loss+water_loss

def loss_996_mse_without_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=40)
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    return sif_loss+lai_loss+agb_loss+nbe_loss

def compute_nse_vec(observed, modeled, mask, epsilon=0.001):
    numerator = jnp.sum((observed - modeled) ** 2 * mask, axis=1)
    observed_mean = jnp.sum(observed * mask, axis=1) / (jnp.sum(mask, axis=1) + epsilon)
    denominator = jnp.sum(((observed.T - observed_mean)**2).T * mask, axis=1) + epsilon
    nse_vec = 1- numerator/denominator
    weighted_nse = jnp.sum(nse_vec * jnp.sum(mask, axis=1)) / (jnp.sum(mask) + epsilon)
    return weighted_nse

def compute_nnse_vec(observed, modeled, mask):
    return 1 / (2 - compute_nse_vec(observed, modeled, mask))

def loss_991_mse_with_grace(params, initial_pools, predictors, met, labels, pfn, batch_forward, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=40)
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)

    predicted_water_pool_mean = jnp.sum((result[:, warm_up:, pfn.next_paw_pool] + result[:, warm_up:, pfn.next_puw_pool]) * labels[:, warm_up:, 7], axis=1) / (jnp.sum(labels[:, warm_up:, 7], axis=1) + 0.01)
    water_loss = patch_of_error(((result[:, warm_up:, pfn.next_paw_pool] + result[:, warm_up:, pfn.next_puw_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)

    return sif_loss+lai_loss+agb_loss+nbe_loss+water_loss

def loss_991_mse_without_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=40)
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    return sif_loss+lai_loss+agb_loss+nbe_loss

def loss_991_nnse_with_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320, epsilon=0.001):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = compute_nnse_vec(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1])
    lai_loss = compute_nnse_vec(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5])
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = compute_nnse_vec(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly)
    nbe_loss = compute_nnse_vec(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3])
    predicted_water_pool_mean = jnp.sum((result[:, warm_up:, pfn.next_paw_pool] + result[:, warm_up:, pfn.next_puw_pool]) * labels[:, warm_up:, 7], axis=1) / (jnp.sum(labels[:, warm_up:, 7], axis=1) + epsilon)
    water_loss = compute_nnse_vec(((result[:, warm_up:, pfn.next_paw_pool] + result[:, warm_up:, pfn.next_puw_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7])

    return -(sif_loss + lai_loss + agb_loss + nbe_loss + water_loss)

def loss_991_nnse_without_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = compute_nnse_vec(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1])
    lai_loss = compute_nnse_vec(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5])
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = compute_nnse_vec(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly)
    nbe_loss = compute_nnse_vec(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3])

    return -(sif_loss + lai_loss + agb_loss + nbe_loss)



def loss_996_nnse_with_grace(params, initial_pools, predictors, met, labels, batch_forward, warm_up=12, batch_size=320, epsilon=0.001):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = compute_nnse_vec(result[:, warm_up:, 39], labels[:, warm_up:, 0], labels[:, warm_up:, 1])
    lai_loss = compute_nnse_vec(result[:, warm_up:, 0], labels[:, warm_up:, 4], labels[:, warm_up:, 5])
    modeled_agb = result[:, warm_up:, 32] + result[:, warm_up:, 33] + result[:, warm_up:, 35]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = compute_nnse_vec(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly)
    nbe_loss = compute_nnse_vec(result[:, warm_up:, 31], labels[:, warm_up:, 2], labels[:, warm_up:, 3])
    predicted_water_pool_mean = jnp.sum(result[:, warm_up:, 38]  * labels[:, warm_up:, 7], axis=1) / (jnp.sum(labels[:, warm_up:, 7], axis=1) + epsilon)
    water_loss = compute_nnse_vec((result[:, warm_up:, 38] .T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7])

    return -(sif_loss + lai_loss + agb_loss + nbe_loss + water_loss)


def loss_996_nnse_without_grace(params, initial_pools, predictors, met, labels, batch_forward, warm_up=12, batch_size=320, epsilon=0.001):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = compute_nnse_vec(result[:, warm_up:, 39], labels[:, warm_up:, 0], labels[:, warm_up:, 1])
    lai_loss = compute_nnse_vec(result[:, warm_up:, 0], labels[:, warm_up:, 4], labels[:, warm_up:, 5])
    modeled_agb = result[:, warm_up:, 32] + result[:, warm_up:, 33] + result[:, warm_up:, 35]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = compute_nnse_vec(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly)
    nbe_loss = compute_nnse_vec(result[:, warm_up:, 31], labels[:, warm_up:, 2], labels[:, warm_up:, 3])

    return -(sif_loss + lai_loss + agb_loss + nbe_loss)

def loss_992_mse_with_grace(params, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320, epsilon=0.001):
    result = batch_forward(params, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=10)
    modeled_biomass = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool] + + result[:, warm_up:, pfn.next_root_pool] 
    observed_biomass = labels[:, warm_up:, 8]
    biomass_mask = labels[:, warm_up:, 9]
    modeled_biomass_monthly = jnp.mean(modeled_biomass.reshape(batch_size, -1, 12), axis=2)
    observed_biomass_monthly=jnp.mean(observed_biomass.reshape(batch_size, -1, 12), axis=2)
    biomass_mask_monthly = jnp.prod(biomass_mask.reshape(biomass_mask.shape[0], -1, 12), axis=2)
    biomass_loss = sum_of_error(modeled_biomass_monthly, observed_biomass_monthly, biomass_mask_monthly, weight=5e-6)

    modeled_som_clim = jnp.mean(result[:, warm_up:, pfn.next_som_pool], axis=1)
    som_loss = sum_of_error(modeled_som_clim, labels[:, 0, 12], labels[:, 0, 13], weight=3e-6)

    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    fire_loss = patch_of_error(result[:, warm_up:, pfn.total_fire_combust], labels[:, warm_up:, 10], labels[:, warm_up:, 11], weight=1000)
    
    violation_loss =  jnp.sum(result[:, :, pfn.violation]) * 400
    predicted_water_pool_mean = jnp.sum((result[:, :, pfn.next_water_pool] ) * labels[:, :, 7], axis=1) / (jnp.sum(labels[:, :, 7], axis=1) + epsilon)
    water_loss = patch_of_error(((result[:, warm_up:, pfn.next_water_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)
    
    return sif_loss + lai_loss + biomass_loss + nbe_loss + violation_loss + fire_loss + som_loss + water_loss


def loss_992_mse_no_cms(params, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320, epsilon=0.001):
    result = batch_forward(params, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=10)
    modeled_biomass = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool] + + result[:, warm_up:, pfn.next_root_pool] 
    observed_biomass = labels[:, warm_up:, 8]
    biomass_mask = labels[:, warm_up:, 9]
    modeled_biomass_monthly = jnp.mean(modeled_biomass.reshape(batch_size, -1, 12), axis=2)
    observed_biomass_monthly=jnp.mean(observed_biomass.reshape(batch_size, -1, 12), axis=2)
    biomass_mask_monthly = jnp.prod(biomass_mask.reshape(biomass_mask.shape[0], -1, 12), axis=2)
    biomass_loss = sum_of_error(modeled_biomass_monthly, observed_biomass_monthly, biomass_mask_monthly, weight=5e-6)

    modeled_som_clim = jnp.mean(result[:, warm_up:, pfn.next_som_pool], axis=1)
    som_loss = sum_of_error(modeled_som_clim, labels[:, 0, 12], labels[:, 0, 13], weight=3e-6)
    
    violation_loss =  jnp.sum(result[:, :, pfn.violation]) * 400
    predicted_water_pool_mean = jnp.sum((result[:, :, pfn.next_water_pool] ) * labels[:, :, 7], axis=1) / (jnp.sum(labels[:, :, 7], axis=1) + epsilon)
    water_loss = patch_of_error(((result[:, warm_up:, pfn.next_water_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)
    
    return sif_loss + lai_loss + biomass_loss + violation_loss + som_loss + water_loss

def loss_993_mse_with_grace(params, predictors, met, labels, batch_forward, pfn, warm_up=0, batch_size=320, epsilon=0.001):
    result = batch_forward(params, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=10)
    modeled_biomass = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool] + + result[:, warm_up:, pfn.next_root_pool] 
    observed_biomass = labels[:, warm_up:, 8]
    biomass_mask = labels[:, warm_up:, 9]
    modeled_biomass_monthly = jnp.mean(modeled_biomass.reshape(batch_size, -1, 12), axis=2)
    observed_biomass_monthly=jnp.mean(observed_biomass.reshape(batch_size, -1, 12), axis=2)
    biomass_mask_monthly = jnp.prod(biomass_mask.reshape(biomass_mask.shape[0], -1, 12), axis=2)
    biomass_loss = sum_of_error(modeled_biomass_monthly, observed_biomass_monthly, biomass_mask_monthly, weight=5e-6)

    modeled_som_clim = jnp.mean(result[:, warm_up:, pfn.next_som_pool], axis=1)
    som_loss = sum_of_error(modeled_som_clim, labels[:, 0, 12], labels[:, 0, 13], weight=3e-6)

    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    fire_loss = patch_of_error(result[:, warm_up:, pfn.total_fire_combust], labels[:, warm_up:, 10], labels[:, warm_up:, 11], weight=1000)
    
    violation_loss =  jnp.sum(result[:, :, pfn.violation]) * 400
    predicted_water_pool_mean = jnp.sum((result[:, :, pfn.next_water_pool] ) * labels[:, :, 7], axis=1) / (jnp.sum(labels[:, :, 7], axis=1) + epsilon)
    water_loss = patch_of_error(((result[:, warm_up:, pfn.next_water_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)
    
    vod_loss = sum_of_error(result[:, warm_up:, pfn.vod], labels[:, warm_up:, 14], labels[:, warm_up:, 15], weight=100)
    
    return sif_loss + lai_loss + biomass_loss + nbe_loss + violation_loss + fire_loss + som_loss + water_loss + vod_loss

def loss_993_mse_with_fluxnet(params, predictors, met, labels, batch_forward, pfn, warm_up=0, batch_size=320, epsilon=0.001):
    result = batch_forward(params, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=10)
    modeled_biomass = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool] + + result[:, warm_up:, pfn.next_root_pool] 
    observed_biomass = labels[:, warm_up:, 8]
    biomass_mask = labels[:, warm_up:, 9]
    modeled_biomass_monthly = jnp.mean(modeled_biomass.reshape(batch_size, -1, 12), axis=2)
    observed_biomass_monthly=jnp.mean(observed_biomass.reshape(batch_size, -1, 12), axis=2)
    biomass_mask_monthly = jnp.prod(biomass_mask.reshape(biomass_mask.shape[0], -1, 12), axis=2)
    biomass_loss = sum_of_error(modeled_biomass_monthly, observed_biomass_monthly, biomass_mask_monthly, weight=5e-6)

    modeled_som_clim = jnp.mean(result[:, warm_up:, pfn.next_som_pool], axis=1)
    som_loss = sum_of_error(modeled_som_clim, labels[:, 0, 12], labels[:, 0, 13], weight=3e-6)

    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    fire_loss = patch_of_error(result[:, warm_up:, pfn.total_fire_combust], labels[:, warm_up:, 10], labels[:, warm_up:, 11], weight=1000)
    
    violation_loss =  jnp.sum(result[:, :, pfn.violation]) * 400
    predicted_water_pool_mean = jnp.sum((result[:, :, pfn.next_water_pool] ) * labels[:, :, 7], axis=1) / (jnp.sum(labels[:, :, 7], axis=1) + epsilon)
    water_loss = patch_of_error(((result[:, warm_up:, pfn.next_water_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)
    
    vod_loss = sum_of_error(result[:, warm_up:, pfn.vod], labels[:, warm_up:, 14], labels[:, warm_up:, 15], weight=100)
    
    fluxnet_gpp_loss = sum_of_error(result[:, warm_up:, pfn.gpp], labels[:, warm_up:, 16], labels[:, warm_up:, 17], weight=500)
    
    fluxnet_reco_loss = sum_of_error(result[:, warm_up:, pfn.gpp] + result[:, warm_up:, pfn.nee],
                                     labels[:, warm_up:, 18], labels[:, warm_up:, 19], weight=500)
    
    fluxnet_et_loss = sum_of_error(result[:, warm_up:, pfn.ET],
                                     labels[:, warm_up:, 20], labels[:, warm_up:, 21], weight=500)
    
    gleam_et_loss = sum_of_error(result[:, warm_up:, pfn.ET],
                                     labels[:, warm_up:, 22], labels[:, warm_up:, 23], weight=1)
    
    return sif_loss + lai_loss + biomass_loss + nbe_loss + violation_loss + fire_loss + som_loss + water_loss + vod_loss + fluxnet_gpp_loss + fluxnet_reco_loss + fluxnet_et_loss + gleam_et_loss


def loss_993_mse_no_cms(params, predictors, met, labels, batch_forward, pfn, warm_up=0, batch_size=320, epsilon=0.001):
    result = batch_forward(params, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=10)
    modeled_biomass = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool] + + result[:, warm_up:, pfn.next_root_pool] 
    observed_biomass = labels[:, warm_up:, 8]
    biomass_mask = labels[:, warm_up:, 9]
    modeled_biomass_monthly = jnp.mean(modeled_biomass.reshape(batch_size, -1, 12), axis=2)
    observed_biomass_monthly=jnp.mean(observed_biomass.reshape(batch_size, -1, 12), axis=2)
    biomass_mask_monthly = jnp.prod(biomass_mask.reshape(biomass_mask.shape[0], -1, 12), axis=2)
    biomass_loss = sum_of_error(modeled_biomass_monthly, observed_biomass_monthly, biomass_mask_monthly, weight=5e-6)

    modeled_som_clim = jnp.mean(result[:, warm_up:, pfn.next_som_pool], axis=1)
    som_loss = sum_of_error(modeled_som_clim, labels[:, 0, 12], labels[:, 0, 13], weight=3e-6)
    
    violation_loss =  jnp.sum(result[:, :, pfn.violation]) * 400
    predicted_water_pool_mean = jnp.sum((result[:, :, pfn.next_water_pool] ) * labels[:, :, 7], axis=1) / (jnp.sum(labels[:, :, 7], axis=1) + epsilon)
    water_loss = patch_of_error(((result[:, warm_up:, pfn.next_water_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)
    vod_loss = sum_of_error(result[:, warm_up:, pfn.vod], labels[:, warm_up:, 14], labels[:, warm_up:, 15], weight=100)

    
    return sif_loss + lai_loss + biomass_loss + violation_loss + som_loss + water_loss + vod_loss

"""
def nse_score(targets, predictions, mask):
    return 1-(np.sum((targets-predictions)**2 * mask)/np.sum((targets-np.sum(targets)/np.sum())**2))

def loss_996_nnse_no_grace(params, initial_pools, predictors, met, labels, batch_forward, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, 39], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=100)
    lai_loss = sum_of_error(result[:, warm_up:, 0], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=5)
    modeled_agb = result[:, warm_up:, 32] + result[:, warm_up:, 33] + result[:, warm_up:, 35]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, 31], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=4000)
    return sif_loss+lai_loss+agb_loss+nbe_loss
"""

# methods for local models
def compute_nse(observed, modeled, mask):
    numerator = jnp.sum((observed - modeled) ** 2 * mask)
    observed_mean = jnp.sum(observed * mask) / jnp.sum(mask)
    denominator = jnp.sum((observed - observed_mean)**2 * mask)
    return 1- numerator/denominator

def compute_nnse(observed, modeled, mask):
    return 1 / (2 - compute_nse(observed, modeled, mask))

def compute_nnse_eval(output_matrix, target_matrix, train_sel, test_sel, reco=False):
        
    gpp_train_nnse = compute_nnse(target_matrix[train_sel, 0], output_matrix[train_sel, 1], target_matrix[train_sel, 1])
    if reco:
        nee_reco_train_nnse = compute_nnse(target_matrix[train_sel, 8], output_matrix[train_sel, 1] + output_matrix[train_sel, 21], target_matrix[train_sel, 9])     
    else:
        nee_reco_train_nnse = compute_nnse(target_matrix[train_sel, 2], output_matrix[train_sel, 21], target_matrix[train_sel, 3])
    et_train_nnse = compute_nnse(target_matrix[train_sel, 4], output_matrix[train_sel, 2], target_matrix[train_sel, 5])
    lai_train_nnse = compute_nnse(target_matrix[train_sel, 6], output_matrix[train_sel, 0], target_matrix[train_sel, 7])
    
    gpp_test_nnse = compute_nnse(target_matrix[test_sel, 0], output_matrix[test_sel, 1], target_matrix[test_sel, 1])
    if reco:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 8], output_matrix[test_sel, 1] + output_matrix[test_sel, 21], target_matrix[test_sel, 9])     
    else:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 2], output_matrix[test_sel, 21], target_matrix[test_sel, 3])
    et_test_nnse = compute_nnse(target_matrix[test_sel, 4], output_matrix[test_sel, 2], target_matrix[test_sel, 5])
    lai_test_nnse = compute_nnse(target_matrix[test_sel, 6], output_matrix[test_sel, 0], target_matrix[test_sel, 7])
    
    return gpp_train_nnse, nee_reco_train_nnse, et_train_nnse, lai_train_nnse, gpp_test_nnse, nee_reco_test_nnse, et_test_nnse, lai_test_nnse

def compute_test_nnse(output_matrix, target_matrix,  test_sel, reco=False):
    gpp_test_nnse = compute_nnse(target_matrix[test_sel, 0], output_matrix[test_sel, 1], target_matrix[test_sel, 1])
    if reco:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 8], output_matrix[test_sel, 1] + output_matrix[test_sel, 21], target_matrix[test_sel, 9])     
    else:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 2], output_matrix[test_sel, 21], target_matrix[test_sel, 3])
    et_test_nnse = compute_nnse(target_matrix[test_sel, 4], output_matrix[test_sel, 2], target_matrix[test_sel, 5])
    lai_test_nnse = compute_nnse(target_matrix[test_sel, 6], output_matrix[test_sel, 0], target_matrix[test_sel, 7])
    return gpp_test_nnse + nee_reco_test_nnse + et_test_nnse + lai_test_nnse

def negative_log_sigmoid(a, b, k=1000000):
    return -jnp.log(1 / (1 + jnp.exp(-k * (a - b))))