import jax.numpy as jnp

def lab_release_factor(t, lab_lifespan, clab_release_period, Bday):
    fl = (jnp.log(lab_lifespan) - jnp.log(lab_lifespan - 1)) * 0.5
    wl = clab_release_period * jnp.sqrt(2) / 2
    osl = offset(lab_lifespan, wl)
    sf = 365.25 / jnp.pi

    return (2 / jnp.sqrt(jnp.pi)) * (fl / wl) * jnp.exp(-(jnp.sin((t - Bday + osl) / sf) * sf / wl)**2)

def leaf_fall_factor(t, leaf_lifespan, leaf_fall_period, Fday):
    ff = (jnp.log(leaf_lifespan) - jnp.log(leaf_lifespan - 1)) * 0.5
    wf = leaf_fall_period * jnp.sqrt(2) / 2
    osf = offset(leaf_lifespan, wf)
    sf = 365.25 / jnp.pi

    return (2 / jnp.sqrt(jnp.pi)) * (ff / wf) * jnp.exp(-(jnp.sin((t - Fday + osf) / sf) * sf / wf)**2)

def offset(L, w):
    p1 = 0.000023599784710
    p2 = 0.000332730053021
    p3 = 0.000901865258885
    p4 = -0.005437736864888
    p5 = -0.020836027517787
    p6 = 0.126972018064287
    p7 = -0.188459767342504

    lf = jnp.log(L - 1)
    os = p1 * lf**6 + p2 * lf**5 + p3 * lf**4 + p4 * lf**3 + p5 * lf**2 + p6 * lf + p7
    os = os * w
    return os
