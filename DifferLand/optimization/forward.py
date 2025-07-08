import jax
import jax.numpy as jnp
from typing import List, Dict

def parameter_prediction_forward(
    params: List[Dict[str, jnp.ndarray]],
    predictors: jnp.ndarray
) -> jnp.ndarray:
    """
    Forward pass through a multi-layer perceptron to produce parameter predictions.

    Each hidden layer applies a linear transformation followed by a leaky ReLU activation.
    The final layer applies only a linear transformation (no activation).

    Parameters
    ----------
    params : list of dict
        A list of layer parameter dictionaries.
        Each dictionary has:
        
        - 'weights' : jax.numpy.ndarray of shape (n_in, n_out)
        - 'biases' : jax.numpy.ndarray of shape (n_out,)
    
    predictors : jax.numpy.ndarray
        Input feature matrix of shape (batch_size, n_features).

    Returns
    -------
    jax.numpy.ndarray
        Output predictions of shape (batch_size, n_outputs).
    """
    *hidden, last = params
    x = predictors
    for layer in hidden:
        x = jax.nn.leaky_relu(x @ layer['weights'] + layer['biases'])
    return x @ last['weights'] + last['biases']


def embed_prediction_forward(
    params: List[Dict[str, jnp.ndarray]],
    predictors: jnp.ndarray
) -> jnp.ndarray:
    """
    Forward pass through a multi-layer perceptron to produce embeddings.

    Each hidden layer applies a linear transformation followed by a leaky ReLU activation.
    The final layer also applies a linear transformation followed by a leaky ReLU activation.

    Parameters
    ----------
    params : list of dict
        A list of layer parameter dictionaries.
        Each dictionary has:
        
        - 'weights' : jax.numpy.ndarray of shape (n_in, n_out)
        - 'biases' : jax.numpy.ndarray of shape (n_out,)
    
    predictors : jax.numpy.ndarray
        Input feature matrix of shape (batch_size, n_features).

    Returns
    -------
    jax.numpy.ndarray
        Embedding output of shape (batch_size, n_outputs).
    """
    *hidden, last = params
    x = predictors
    for layer in hidden:
        x = jax.nn.leaky_relu(x @ layer['weights'] + layer['biases'])
    return jax.nn.leaky_relu(x @ last['weights'] + last['biases'])


