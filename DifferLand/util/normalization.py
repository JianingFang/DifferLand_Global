import jax.numpy as jnp

def _nor2par(p, mn, mx):
    return mn * (mx/mn)**p

def nor2par(x, mn, mx):
    return _nor2par(jnp.arctan(x)/jnp.pi+0.5, mn, mx)

def _par2nor(p, mn, mx):
    return jnp.log(p/mn)/jnp.log(mx/mn)

def par2nor(x, mn, mx):
    return jnp.tan((_par2nor(x, mn, mx)-0.5)*jnp.pi)


def unnormalize_parameters(normalized_parameters, param_parmin, param_parmax):
    return nor2par(normalized_parameters, param_parmin, param_parmax)