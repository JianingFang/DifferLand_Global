import jax.numpy as jnp

def ACM(lat, doy, t_max, t_min, lai, rad, ca, ce):
    d1 = 0.0156935  # day length constant
    θ = 4.22273
    k = 208.868
    d2 = 0.0453194
    b2 = 0.37836
    c1 = 7.19298  # Fixed the typo 'a' to '*'
    a2 = 0.011136
    c2 = 2.1001
    eb1T = 0.789798
    ψ_d = -2
    H = 1

    # compute daily canopy conductance, gc
    gc = jnp.abs(ψ_d)**eb1T / (b2 * H + 0.5 * (t_max - t_min))

    # compute p parameter needed for ci
    p = lai * 1 * ce * jnp.exp(a2 * t_max) / gc

    # compute the q parameter needed for ci
    q = θ - k

    # compute the internal CO2 concentration, ci
    ci = 0.5 * (ca + q - p + jnp.sqrt((ca + q - p)**2 - 4 * (ca * q - p * θ)))

    # compute canopy-level quantum yield, e0
    e0 = c1 * (lai**2) / (c2 + lai**2)

    dec = -23.4 * jnp.cos((360 * (doy + 10) / 365) * jnp.pi / 180) * jnp.pi / 180
    mult = jnp.tan(lat * jnp.pi / 180) * jnp.tan(dec)
    mult_valid = (mult < 1) * (mult > -1)
    mult_temp = mult * mult_valid
    dayl = 24 * jnp.arccos(-mult_temp) / jnp.pi
    mult_geq_one_sel = (mult < 1)
    dayl = dayl * mult_geq_one_sel + (1 - mult_geq_one_sel) * 24
    mult_leq_minus_one_sel = (mult > -1)
    dayl = dayl * mult_leq_minus_one_sel + (1 - mult_geq_one_sel) * 0

    # compute co2 rate of diffusion to the site of fixation, pd
    pd = gc * (ca - ci)

    # compute light limitation pi
    pi = e0 * rad * pd / (e0 * rad + pd)

    # compute gpp
    gpp = pi * (d1 * dayl + d2)
    
    return gpp