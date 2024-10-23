import jax


def parameter_prediction_forward(params, predictors):
    *hidden, last = params
    x = predictors
    for layer in hidden:
        x = jax.nn.leaky_relu(x @ layer['weights'] + layer['biases'])
    return x @ last['weights'] + last['biases']

def embed_prediction_forward(params, predictors):
    *hidden, last = params
    x = predictors
    for layer in hidden:
        x = jax.nn.leaky_relu(x @ layer['weights'] + layer['biases'])
    return jax.nn.leaky_relu(x @ last['weights'] + last['biases'])


