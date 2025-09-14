.. _compute_shap:

Ensemble SHAP Analysis
======================

Overview
--------

The script ``compute_shap.py`` is used to compute **ensemble SHAP values** for
DifferLand’s spatialization network. SHAP (SHapley Additive exPlanations) is
applied to quantify the marginal contribution of each environmental predictor
to the learned ecological parameters. By using the average of an ensemble of model members, the
procedure reduces random uncertainties associated with individual neural network
initializations.

Methodology
-----------

We applied **Kernel SHAP** to extract the environment–parameter relationships
learned by the spatialization neural network. Kernel SHAP estimates Shapley
values, informed by cooperative game theory, to quantify each feature’s
contribution to the model output.

To ensure robustness of SHAP results:

* We defined an **ensemble-averaged model** :math:`M_{ens}` from **10 out of 20**
  model members for each configuration that best converged on the training set:

  .. math::

     M_{ens} = \frac{1}{10} \sum_{i=1}^{n} M(x, \theta_i)

  where :math:`M` is the spatialization network, :math:`x` are spatial
  predictors, and :math:`\theta_i` are the neural network weights and biases of
  member :math:`i`. Typically we have n=10.

* From :math:`M_{ens}`, **100 grid cells** were sampled to define the background
  distribution.  
* SHAP values were computed on **1000 randomly selected grid cells** within the
  training dataset.  
* **Feature importance** was derived by ranking the mean absolute SHAP values
  across the 1000 samples.  

This procedure is repeated for each selected ecological parameter, providing
global trait–environment relationships.

Conditional SHAP by PFT
-----------------------

In addition to global analysis, **conditional SHAP values** can be computed for
specific plant functional types (PFTs). This is done by restricting the sampled
grid cells to those where a given PFT constitutes at least 80% of the area.  
This enables evaluation of **ecosystem-specific trait–environment
relationships**.

Outputs
-------

The script produces:

* **Global SHAP values**: Feature importances for one ecological parameter at a
  time.  
* **Conditional SHAP values**: Predictor contributions within specific plant
  functional types.  

Usage
-----

A typical workflow involves specifying the ensemble member indices and a single
target parameter for SHAP analysis. For example, to compute unconditioned relationships:

.. code-block:: bash

   python compute_shap.py \
     --run 1,2,3,4,5,6,7,8,9,10 \
     --predictors CLIM+SOIL+AGE \
     --target canopy_efficiency

Or to condition on a specific plant functional type:

.. code-block:: bash

   python compute_shap.py \
     --run 1,2,3,4,5,6,7,8,9,10 \
     --predictors PFT+CLIM+SOIL+AGE \
     --target canopy_efficiency \
     --pft GRA

This will:

1. Load the checkpoints for the specified members.
2. Build the ensemble-averaged model.
3. Compute SHAP values for the specified parameter.
4. Save SHAP arrays in the corresponding directory. These can be later accessed to generate SHAP plots.
