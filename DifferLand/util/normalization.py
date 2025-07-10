import jax.numpy as jnp
from jax import Array
from typing import Union


def _nor2par(
    p: Union[float, Array], mn: Union[float, Array], mx: Union[float, Array]
) -> Union[float, Array]:
    """
    Convert a normalized parameter in [0, 1] to its physical value.

    Computes:

        param = mn * (mx / mn) ** p

    Parameters
    ----------
    p : float or Array
        Normalized parameter in the interval [0, 1].
    mn : float or Array
        Minimum physical bound of the parameter.
    mx : float or Array
        Maximum physical bound of the parameter.

    Returns
    -------
    param : float or Array
        Physical parameter value in the range [mn, mx].
    """
    return mn * (mx / mn) ** p


def nor2par(
    x: Union[float, Array], mn: Union[float, Array], mx: Union[float, Array]
) -> Union[float, Array]:
    """
    Convert a parameter from an unbounded normalized space (−∞, ∞) to its physical value.

    Maps a real-valued normalized variable `x` into [0, 1] via:

        p = arctan(x) / π + 0.5

    and then scales it to [mn, mx].

    Parameters
    ----------
    x : float or Array
        Normalized parameter in (−∞, ∞).
    mn : float or Array
        Minimum physical bound of the parameter.
    mx : float or Array
        Maximum physical bound of the parameter.

    Returns
    -------
    param : float or Array
        Physical parameter value in the range [mn, mx].
    """
    return _nor2par(jnp.arctan(x) / jnp.pi + 0.5, mn, mx)


def _par2nor(
    p: Union[float, Array], mn: Union[float, Array], mx: Union[float, Array]
) -> Union[float, Array]:
    """
    Convert a physical parameter in [mn, mx] to a normalized value in [0, 1].

    Computes:

        p_norm = log(p / mn) / log(mx / mn)

    Parameters
    ----------
    p : float or Array
        Physical parameter value in the range [mn, mx].
    mn : float or Array
        Minimum physical bound of the parameter.
    mx : float or Array
        Maximum physical bound of the parameter.

    Returns
    -------
    p_norm : float or Array
        Normalized parameter in [0, 1].
    """
    return jnp.log(p / mn) / jnp.log(mx / mn)


def par2nor(
    x: Union[float, Array], mn: Union[float, Array], mx: Union[float, Array]
) -> Union[float, Array]:
    """
    Convert a physical parameter value to an unbounded normalized variable (−∞, ∞).

    Maps the physical parameter into [0, 1], then transforms it via:

        norm = tan((p_norm − 0.5) * π)

    Parameters
    ----------
    x : float or Array
        Physical parameter value in the range [mn, mx].
    mn : float or Array
        Minimum physical bound of the parameter.
    mx : float or Array
        Maximum physical bound of the parameter.

    Returns
    -------
    norm : float or Array
        Normalized parameter in (−∞, ∞).
    """
    return jnp.tan((_par2nor(x, mn, mx) - 0.5) * jnp.pi)


def unnormalize_parameters(
    normalized_parameters: Array,
    param_parmin: Array,
    param_parmax: Array,
) -> Array:
    """
    Convert normalized parameters back to physical values.

    This function converts normalized parameters from an unbounded space
    (−∞, ∞) to their physical ranges [min, max] using logarithmic scaling.

    Parameters
    ----------
    normalized_parameters : Array
        Normalized parameter(s) in (−∞, ∞).
    param_parmin : Array
        Minimum physical bound(s) for the parameters.
    param_parmax : Array
        Maximum physical bound(s) for the parameters.

    Returns
    -------
    parameters : Array
        Physical parameter values in their bounded ranges.
    """
    return nor2par(normalized_parameters, param_parmin, param_parmax)
