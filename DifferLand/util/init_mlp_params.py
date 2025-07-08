import jax
import jax.numpy as jnp
from typing import List, Dict

def init_mlp_params(layer_widths: List[int], n: int = 42) -> List[Dict[str, jnp.ndarray]]:
    """
    Initialize parameters for a multi-layer perceptron (MLP) using Glorot normal initialization.

    Parameters
    ----------
    layer_widths : list of int
        List specifying the number of neurons in each layer of the MLP.
        For example, [4, 10, 1] defines a network with input size 4,
        one hidden layer of size 10, and output size 1.
    n : int, optional
        Random seed for JAX PRNGKey (default is 42).

    Returns
    -------
    list of dict
        A list of dictionaries, one per layer, each containing:
        
        - 'weights' : jax.numpy.ndarray
            Weight matrix of shape (n_in, n_out).
        - 'biases' : jax.numpy.ndarray
            Bias vector of shape (n_out,).

    Examples
    --------
    >>> params = init_mlp_params([3, 5, 2])
    >>> for layer in params:
    ...     print(layer["weights"].shape, layer["biases"].shape)
    (3, 5) (5,)
    (5, 2) (2,)
    """
    key = jax.random.PRNGKey(n)
    initializer = jax.nn.initializers.glorot_normal()
    params = []

    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        key, subkey = jax.random.split(key)
        weights = initializer(subkey, (n_in, n_out), jnp.float32)
        biases = jnp.ones(shape=(n_out,), dtype=jnp.float32)
        params.append({
            'weights': weights,
            'biases': biases
        })

    return params
