import jax
import jax.numpy as jnp

def init_mlp_params(layer_widths, n=42):
    key=jax.random.PRNGKey(n)
    initializer = jax.nn.initializers.glorot_normal()
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        initializer = jax.nn.initializers.glorot_normal()
        new_key, subkey = jax.random.split(key)
        del key
        params.append(dict(weights=initializer(subkey, (n_in, n_out), jnp.float32) , biases=jnp.ones(shape=(n_out,))))
        del subkey
        key = new_key
    return params