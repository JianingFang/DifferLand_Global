from dataclasses import dataclass, field
from DifferLand.util.normalization import unnormalize_parameters
import jax

@dataclass
class DALECBase:
    """
    Base class for the DALEC model framework.

    This class defines the interface for DALEC models, including the required
    methods to be implemented in any derived class.

    Methods
    -------
    forward(``*args``, ``**kwargs``)
        Run the model forward in time.

    unnormalize_parameters(``*args``, ``**kwargs``)
        Convert normalized parameters back to their original scale.

    step(``*args``, ``**kwargs``)
        Perform a single time step of the model.
    """
    
    param_parmin: jax.Array = field(init=False)
    param_parmax: jax.Array = field(init=False)
    pool_parmax: jax.Array = field(init=False)
    pool_parmin: jax.Array = field(init=False)
    id: jax.Array = field(init=False)

    def __init__(self):
        """
        Initialize the DALEC model instance.

        Any specific initialization for derived classes should be implemented
        in subclasses.
        """
        pass

    def forward(self, *args, **kwargs):
        """
        Run the model forward in time.

        Parameters
        ----------
        ``*args`` :
            Positional arguments for the forward model run.
        ``**kwargs`` :
            Keyword arguments for the forward model run.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError


    def step(self, *args, **kwargs):
        """
        Perform a single time step of the model.

        Parameters
        ----------
        ``*args`` :
            Positional arguments for the model step.
        ``**kwargs`` :
            Keyword arguments for the model step.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def unnormalize(self, normalized_parameters):
        return unnormalize_parameters(normalized_parameters, param_parmin=self.param_parmin, param_parmax=self.param_parmax)
    
    
    def unnormalize_pheno(self, normalized_pools):
        return unnormalize_parameters(normalized_pools, param_parmin=self.pheno_parmin, param_parmax=self.pheno_parmax)
    
    
    