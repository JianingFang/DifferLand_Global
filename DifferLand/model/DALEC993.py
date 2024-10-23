import jax.numpy as jnp
import jax
from functools import partial
from DifferLand.model.auxi.phenology import leaf_fall_factor, lab_release_factor
from DifferLand.model.auxi.ACM import ACM
from DifferLand.model.DALEC_993_parinfo import dalec993_parmax, dalec993_parmin, dalec993_param_parmin, dalec993_param_parmax, dalec993_pfn
from DifferLand.model.DALEC_993_parinfo import dalec993_pool_parmax, dalec993_pool_parmin
from DifferLand.model.DALEC_993_parinfo import dalec993_pheno_parmax, dalec993_pheno_parmin

from DifferLand.optimization.forward import parameter_prediction_forward
from DifferLand.model.DALEC import DALEC
from DifferLand.util.normalization import unnormalize_parameters

class DALEC993(DALEC):
    def __init__(self, water_stress_type="default"):
        super().__init__()
        self.parmin = dalec993_parmin
        self.parmax = dalec993_parmax
        self.param_parmin = dalec993_param_parmin
        self.param_parmax = dalec993_param_parmax
        self.water_stress_type = water_stress_type
        self.pfn = dalec993_pfn
        self.pool_parmax = dalec993_pool_parmax
        self.pool_parmin = dalec993_pool_parmin
        self.pheno_parmax = dalec993_pheno_parmax
        self.pheno_parmin = dalec993_pheno_parmin
        self.id = 993
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, pools, met, dalec_parameters, pheno_parameters, gpp_params=None):
        time = met[0]
        t_min = met[1]
        t_max = met[2]
        rad = met[3]
        ca = met[4]
        doy = met[5]
        burned_area = met[6]
        vpd = met[7]
        precipitation = met[8]
        lat = met[9]
        delta_t = met[10]
        t_mean = met[11]
        mean_precipitation = met[12]
        norm_temp = met[13]
        norm_solar = met[14]
        norm_vpd = met[15]
        norm_ca = met[16]
        
        labile_pool = pools[0]
        foliar_pool = pools[1]
        root_pool = pools[2]
        wood_pool = pools[3]
        litter_pool = pools[4]
        som_pool = pools[5]
        water_pool = pools[6]
                
        
        decomposition_rate = dalec_parameters[0]  # decomposition rate
        f_auto = dalec_parameters[1]  # fraction of GPP respired
        f_fol = dalec_parameters[2]  # fraction of (1-fpp) to foliage
        f_root = dalec_parameters[3]  # fraction of (1-fgpp) to roots
        leaf_lifespan = dalec_parameters[4]  # leaf lifespan
        tor_wood = dalec_parameters[5]  # turn over rate wood - 1% loss per year value
        tor_root = dalec_parameters[6]  # turn over rate root
        tor_litter = dalec_parameters[7]  # TOR litter
        tor_som = dalec_parameters[8]  # TOR SOM
        Q10 = dalec_parameters[9]  # Temp factor

        ce = dalec_parameters[10]  # Canopy Efficiency (time invariant)
        Bday = pheno_parameters[0]  # Bday
        f_lab = dalec_parameters[11]  # Fraction to clab
        clab_release_period = dalec_parameters[12]  # Clab release period
        Fday = pheno_parameters[1]  # Fday
        leaf_fall_period = dalec_parameters[13]  # leaf fall period
        LCMA = dalec_parameters[14]  # Leaf carbon mass per area
        IWUE = dalec_parameters[15]  # IWUE: GPP*VPD/ET: gC/kgH2o *hPa
        runoff_focal_point = dalec_parameters[16]  # Runoff focal point (~maximum soil storage capacity x 4)
        field_capacity = dalec_parameters[17]
        foliar_cf = dalec_parameters[18]  # foliar biomass cf
        ligneous_cf = dalec_parameters[19]  # ligneous biomass cf
        dom_cf = dalec_parameters[20]  # dom cf
        resilience = dalec_parameters[21]  # resilience factor
        lab_lifespan = dalec_parameters[22]  # labile pool lifespan
        moisture_factor = dalec_parameters[23]  # moisture factor
        uWUE = dalec_parameters[24] # boese_r for ET correction
        boese_r = dalec_parameters[25] # boese_r for ET correction
        wilting_point_frac = dalec_parameters[26] # wilting point fraction as of field capacity
        sif_alpha = dalec_parameters[27]  # alpha param for SIF-GPP relationship
        sif_beta_plus_three = dalec_parameters[28]  # beta param for SIF-GPP relationship
        sif_beta = sif_beta_plus_three - 3.0
        p_fol = dalec_parameters[29]  # foliar pool coefficient for VOD assimilation
        p_wood = dalec_parameters[30] # wood pool coefficient for VOD assimilation
  
        
        lai= foliar_pool/LCMA
        
        if self.water_stress_type == "nothing":
            beta = 1
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "default":        
            wilting_point =field_capacity * wilting_point_frac
            beta = (water_pool - wilting_point) / (field_capacity - wilting_point)
            beta = jnp.where(beta <=1, beta, 1)
            beta = jnp.where(beta >=0, beta, 0)
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "nn_paw":
            beta = jax.nn.sigmoid(parameter_prediction_forward(gpp_params, jnp.array([water_pool/1500,]))[0])
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "nn_whole":
            beta = -9999
            raw_gpp, raw_ET = parameter_prediction_forward(gpp_params, jnp.array([norm_temp, lai/8, norm_solar,  water_pool/1500, norm_vpd, norm_ca]))
            gpp = jnp.maximum(0.01 * raw_gpp , raw_gpp)
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)
        elif self.water_stress_type == "nn_whole_no_lai":
            beta = -9999
            raw_gpp, raw_ET = parameter_prediction_forward(gpp_params, jnp.array([norm_temp, norm_solar,  water_pool/1500, norm_vpd, norm_ca]))
            gpp = jnp.maximum(0.01 * raw_gpp , raw_gpp)
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)
        elif self.water_stress_type == "gpp_acm_et_nn":
            beta = -9999
            beta_params, et_params = gpp_params
            beta = jax.nn.sigmoid(parameter_prediction_forward(beta_params, jnp.array([water_pool/1500,]))[0])
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            raw_ET = parameter_prediction_forward(et_params, jnp.array([norm_temp, lai/8, norm_solar,  water_pool/1500, norm_vpd, norm_ca, gpp/8]))[0]
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)

        SIF = (1.0 / sif_alpha) * gpp - sif_beta / sif_alpha

        temperate = (jnp.exp(Q10 * (0.5 * (t_max + t_min) - t_mean)) 
                    * ((precipitation / mean_precipitation - 1) * moisture_factor + 1))

        respiration_auto = f_auto * gpp
        leaf_production = (gpp - respiration_auto) * f_fol
        labile_production = (gpp - respiration_auto - leaf_production) * f_lab
        root_production = (gpp - respiration_auto - leaf_production - labile_production) * f_root
        wood_production = gpp - respiration_auto - leaf_production - labile_production - root_production

        lff = leaf_fall_factor(time, leaf_lifespan, leaf_fall_period, Fday)
        lrf = lab_release_factor(time, lab_lifespan, clab_release_period, Bday)

        labile_release = labile_pool * (1 - (1 - lrf) ** delta_t) / delta_t
        leaf_litter = foliar_pool * (1 - (1 - lff) ** delta_t) / delta_t
        wood_litter = wood_pool * (1 - (1 - tor_wood) ** delta_t) / delta_t
        root_litter = root_pool * (1 - (1 - tor_root) ** delta_t) / delta_t
        
        respiration_hetero_litter = litter_pool * (1 - (1 - temperate * tor_litter) ** delta_t) / delta_t
        respiration_hetero_som = som_pool * (1 - (1 - temperate * tor_som) ** delta_t) / delta_t
        litter_to_som = litter_pool * (1 - (1 - temperate * decomposition_rate) ** delta_t) / delta_t
        
        runoff = water_pool ** 2 / runoff_focal_point / delta_t
        runoff = jnp.where(water_pool > runoff_focal_point / 2, (water_pool - runoff_focal_point / 4) / delta_t, runoff)
        
        next_labile_pool = labile_pool + (labile_production - labile_release) * delta_t
        Clab_min_sel = next_labile_pool >= self.parmin.Clab
        next_labile_pool = jnp.where(Clab_min_sel, next_labile_pool, self.parmin.Clab)
        labile_release = jnp.where(Clab_min_sel, labile_release, labile_production - (next_labile_pool - labile_pool) / delta_t)
        Clab_max_sel = next_labile_pool <= self.parmax.Clab
        next_labile_pool = jnp.where(Clab_max_sel, next_labile_pool, self.parmax.Clab)
        labile_release = jnp.where(Clab_max_sel, labile_release, labile_production - (next_labile_pool - labile_pool) / delta_t)
        
        next_foliar_pool = foliar_pool + (leaf_production - leaf_litter + labile_release) * delta_t
        Cfol_min_sel = next_foliar_pool >= self.parmin.Cfol
        next_foliar_pool = jnp.where(Cfol_min_sel, next_foliar_pool, self.parmin.Cfol)
        leaf_litter = jnp.where(Cfol_min_sel, leaf_litter, (1 - Cfol_min_sel) * self.parmin.Cfol)
        Cfol_max_sel = next_foliar_pool <= self.parmax.Cfol
        next_foliar_pool = jnp.where(Cfol_max_sel, next_foliar_pool, self.parmax.Cfol)
        leaf_litter = jnp.where(Cfol_max_sel, leaf_litter, leaf_production + labile_release - (next_foliar_pool - foliar_pool) / delta_t)

        next_root_pool = root_pool + (root_production - root_litter) * delta_t
        Croot_min_sel = next_root_pool >= self.parmin.Croot
        next_root_pool = jnp.where(Croot_min_sel, next_root_pool, self.parmin.Croot)
        root_litter = jnp.where(Croot_min_sel, root_litter, root_production - (next_root_pool - root_pool) / delta_t)
        Croot_max_sel = next_root_pool <= self.parmax.Croot
        next_root_pool = jnp.where(Croot_max_sel, next_root_pool, self.parmax.Croot)
        root_litter = jnp.where(Croot_max_sel, root_litter, root_production - (next_root_pool - root_pool) / delta_t)
        
        next_wood_pool = wood_pool + (wood_production - wood_litter) * delta_t
        Cwood_min_sel = next_wood_pool >= self.parmin.Cwood
        next_wood_pool = jnp.where(Cwood_min_sel, next_wood_pool, self.parmin.Cwood)
        wood_litter = jnp.where(Cwood_min_sel, wood_litter, wood_production - (next_wood_pool - wood_pool) / delta_t)
        Cwood_max_sel = next_wood_pool <= self.parmax.Cwood
        next_wood_pool = jnp.where(Cwood_max_sel, next_wood_pool, self.parmax.Cwood)
        wood_litter = jnp.where(Cwood_max_sel, wood_litter, wood_production - (next_wood_pool - wood_pool) / delta_t)

        
        next_litter_pool = litter_pool + (leaf_litter + root_litter - respiration_hetero_litter - litter_to_som) * delta_t
        Clitter_min_sel = next_litter_pool >= self.parmin.Clitter
        next_litter_pool = jnp.where(Clitter_min_sel, next_litter_pool, self.parmin.Clitter)
        litter_to_som = jnp.where(Clitter_min_sel, litter_to_som, leaf_litter + root_litter - respiration_hetero_litter - (next_litter_pool - litter_pool) / delta_t)
        litter_to_som_sel = litter_to_som >= 0
        litter_to_som = jnp.where(litter_to_som_sel, litter_to_som, 0)
        respiration_hetero_litter = jnp.where(litter_to_som_sel, respiration_hetero_litter, leaf_litter + root_litter - (next_litter_pool - litter_pool) / delta_t)
        Clitter_max_sel = next_litter_pool <= self.parmax.Clitter
        next_litter_pool = jnp.where(Clitter_max_sel, next_litter_pool, self.parmax.Clitter)
        litter_to_som = jnp.where(Clitter_max_sel, litter_to_som, leaf_litter + root_litter - respiration_hetero_litter - (next_litter_pool - litter_pool) / delta_t)

        next_som_pool = som_pool + (litter_to_som - respiration_hetero_som + wood_litter) * delta_t
        Csom_min_sel = next_som_pool >= self.parmin.Csom
        next_som_pool = jnp.where(Csom_min_sel, next_som_pool, self.parmin.Csom)
        respiration_hetero_som = jnp.where(Csom_min_sel, respiration_hetero_som, litter_to_som + wood_litter - (next_som_pool - som_pool) / delta_t)
        Csom_max_sel = next_som_pool <= self.parmax.Csom
        next_som_pool = jnp.where(Csom_max_sel, next_som_pool, self.parmax.Csom)
        respiration_hetero_som = jnp.where(Csom_max_sel, respiration_hetero_som, litter_to_som + wood_litter - (next_som_pool - som_pool) / delta_t)
        
        next_water_pool = water_pool - runoff * delta_t + precipitation * delta_t - ET * delta_t
        water_min_sel = next_water_pool >= self.parmin.initial_water
        next_water_pool = water_min_sel * next_water_pool + (1 - water_min_sel) *  self.parmin.initial_water
        runoff = jnp.where(water_min_sel, runoff, precipitation  - ET  - (next_water_pool - water_pool) / delta_t)
        run_off_sel = runoff >= 0.0
        violation = jnp.maximum(-runoff * 0.01, 0)
        ET = jnp.where(run_off_sel, ET, precipitation - (next_water_pool - water_pool) / delta_t)
        water_max_sel = next_water_pool <= self.parmax.initial_water
        next_water_pool =  jnp.where(water_max_sel, next_water_pool, self.parmax.initial_water)
        runoff = jnp.where(water_max_sel, runoff, precipitation  - ET  - (next_water_pool - water_pool) / delta_t)

        labile_fire_combust = next_labile_pool*burned_area * ligneous_cf / delta_t
        foliar_fire_combust = next_foliar_pool*burned_area * foliar_cf / delta_t
        root_fire_combust = next_root_pool*burned_area * ligneous_cf / delta_t
        wood_fire_combust = next_wood_pool*burned_area * ligneous_cf / delta_t
        litter_fire_combust = next_litter_pool*burned_area * (ligneous_cf + foliar_cf) * 0.5 /delta_t
        som_fire_combust = next_som_pool*burned_area * dom_cf / delta_t

        labile_fire_transfer = next_labile_pool * burned_area*(1-ligneous_cf) * (1-resilience) / delta_t
        foliar_fire_transfer = next_foliar_pool * burned_area*(1-foliar_cf) * (1-resilience) /delta_t
        root_fire_transfer = next_root_pool * burned_area * (1-ligneous_cf) * (1-resilience)/delta_t
        wood_fire_transfer = next_wood_pool * burned_area*(1-ligneous_cf) * (1-resilience)/delta_t
        litter_fire_transfer = next_litter_pool * burned_area*(1-(ligneous_cf+foliar_cf) * 0.5)*(1-resilience)/delta_t

        next_labile_pool = next_labile_pool-(labile_fire_combust+labile_fire_transfer)*delta_t
        next_foliar_pool = next_foliar_pool-(foliar_fire_combust+foliar_fire_transfer)*delta_t
        next_root_pool = next_root_pool-(root_fire_combust+root_fire_transfer)*delta_t
        next_wood_pool = next_wood_pool-(wood_fire_combust+wood_fire_transfer)*delta_t
        next_litter_pool = next_litter_pool+(labile_fire_transfer+foliar_fire_transfer+root_fire_transfer-litter_fire_combust-litter_fire_transfer)*delta_t
        next_som_pool = next_som_pool+(wood_fire_transfer+litter_fire_transfer-som_fire_combust)*delta_t

        total_fire_combust = labile_fire_combust + foliar_fire_combust + root_fire_combust + wood_fire_combust + litter_fire_combust + som_fire_combust

        nee = -gpp + respiration_auto + respiration_hetero_litter + respiration_hetero_som - total_fire_combust
        
        vod = p_fol * next_foliar_pool + p_wood * next_wood_pool
        
        new_pools = jnp.array([next_labile_pool, next_foliar_pool,
                            next_root_pool, next_wood_pool, next_litter_pool,
                            next_som_pool, next_water_pool])
        
        
        new_fluxes = jnp.array([[lai, gpp, ET, temperate, respiration_auto, leaf_production, labile_production, root_production,
    wood_production, lff, lrf, labile_release, leaf_litter, wood_litter, root_litter, respiration_hetero_litter,
    respiration_hetero_som, litter_to_som, runoff,labile_fire_combust, foliar_fire_combust, root_fire_combust,
    wood_fire_combust, litter_fire_combust, som_fire_combust, labile_fire_transfer, foliar_fire_transfer,
    root_fire_transfer, wood_fire_transfer, litter_fire_transfer, total_fire_combust, 
    nee, next_labile_pool, next_foliar_pool,
    next_root_pool, next_wood_pool, next_litter_pool, next_som_pool, next_water_pool, beta, SIF, violation, vod]])
        
        return new_pools, new_fluxes
        
    @partial(jax.jit, static_argnums=(0,))
    def unnormalize(self, normalized_parameters):
        return unnormalize_parameters(normalized_parameters, param_parmin=self.param_parmin, param_parmax=self.param_parmax)
    
    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_water(self, normalized_parameters):
        return unnormalize_parameters(normalized_parameters, jnp.array([1.0e0, ]), param_parmax=jnp.array([10000e0,]))
    
    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_pools(self, normalized_pools):
        return unnormalize_parameters(normalized_pools, param_parmin=self.pool_parmin, param_parmax=self.pool_parmax)
    
    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_pheno(self, normalized_pools):
        return unnormalize_parameters(normalized_pools, param_parmin=self.pheno_parmin, param_parmax=self.pheno_parmax)