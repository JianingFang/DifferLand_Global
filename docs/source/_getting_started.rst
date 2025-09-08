Tips for Building Your Model
============================

DifferLand is designed to be modular and extensible. Users can implement new
process representations, data sources, and loss terms to tailor the framework
to their research needs. Below are recommended practices for building your own
model extensions.

Extending DALEC Models
----------------------

* Implement a new DALEC model version in ``DALEC[MODELID].py`` by extending the
  ``DALECBase`` class in ``DifferLand.model``.
* Use ``DALEC993.py`` as an example implementation.
* Define the physical boundaries of new parameters and specify any new modeled
  variables in a corresponding ``DALEC_[MODELID]_parinfo.py`` file.

Adding New Data
---------------

* New datasets can be incorporated by writing them into the
  ``differland_global_driver_vX.nc`` file.
* If your new driver data contains **spatial gaps**, you must update the
  corresponding ``era_valid_vX.nc`` file to mask out invalid pixels. This
  prevents propagation of NaN values during training.
* By design, all **meteorological drivers** and **spatial predictors** are
  expected to be **temporally gap-free**.
* Training targets may contain gaps. The data loading logic in
  ``experiments/calibration.py`` can handle missing data through masking.

Customizing Loss Functions
--------------------------

* Additional loss terms can be implemented in ``loss_functions.py``.
* To integrate new data into training, update the data loading logic in
  ``experiments/calibration.py`` so that the new variables are available during
  model calibration.

Best Practices
--------------

* **Numerical stability**: Be mindful of potential issues such as exploding or
  vanishing values. Debug with tools like ``jax.debug`` to locate the source of
  NaN values during development.
* **Equifinality**: Introducing new processes can increase parameter redundancy.
  To avoid this, proceed incrementally—validate each modification before adding
  more complexity.
* **Incremental development**: Start with a minimal working model and expand
  step by step. Confirm each new addition produces stable and interpretable
  outputs before moving forward.

By following these guidelines, you can extend DifferLand while maintaining
model interpretability, stability, and performance.
