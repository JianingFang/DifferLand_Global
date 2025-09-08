Known Limitations
=================

Despite strong overall performance, the current hybrid DALEC framework has
structural limitations that constrain predictive skill in certain processes.

Water Cycle
-----------

* **Simplified ET formulation**: Based on fixed water-use efficiency, which does
  not capture dynamic stomatal regulation under variable meteorological
  conditions.
* **Missing processes**:  
  - GRACE-derived EWT includes deep groundwater and large-scale hydrology not represented in DALEC’s soil-bucket scheme.  
  - Vertical soil moisture transport and groundwater dynamics are absent.
* **Potential improvements**: Incorporating mechanistic stomatal and
  transpiration formulations (possibly hybrid-ML approaches), multi-layer soil
  hydrology distinguishing plant-available versus unavailable water, and
  watershed-scale constraints such as streamflow.

Phenology
---------

* **Fixed leaf dynamics**: DALEC prescribes static onset dates and durations for
  leaf growth and senescence.
* **Limitations**:  
  - Cannot capture interannual variability in leaf dynamics driven by temperature, soil moisture, vapor pressure deficit, or photoperiod.  
  - Produces realistic mean seasonal cycles of LAI, but does not fully reproduce observed interannual variability in climate-sensitive ecosystems.
* **Potential improvements**: More flexible formulations that couple
  mechanistic environmental cues with data-driven parameterizations, and
  explicit representation of leaf turnover and age structure.

Land Use and Land Cover Change
------------------------------

* **Not explicitly represented**: DifferLand currently assumes static land cover
  and does not account for land use transitions such as deforestation,
  afforestation, cropland expansion, or urbanization.
* **Implications**: This limits applicability for studies where anthropogenic
  land use change is a major driver of carbon and water fluxes.
* **Potential improvements**: Coupling with land use change datasets or adding
  dynamic land cover modules to better capture anthropogenic impacts.

Nutrient Limitation
-------------------

* **No explicit nutrient cycles**: The model does not include nitrogen or
  phosphorus cycling.
* **Implications**: Ecosystem processes such as photosynthesis, carbon
  allocation, and decomposition are simulated without nutrient constraints.
  This may overestimate productivity in nutrient-limited ecosystems (e.g.,
  tropical forests, boreal soils).
* **Potential improvements**: Incorporating simplified or hybrid-ML nutrient
  limitation schemes, or coupling with existing nutrient cycle modules, to
  represent N and P constraints on carbon dynamics.


Future Improvements
-------------------

These limitations highlight **priority areas for future hybrid land model
development**, particularly:

* More mechanistic yet flexible hydrology  
* Improved phenology formulations  
* Explicit representation of land use and land cover change  
* Inclusion of nutrient cycling (N, P)  
* Stronger coupling with diverse observational constraints  

Such improvements are needed to better capture ecosystem responses under a
changing climate.

Also, if you found a bug 🐛 or would like to suggest an improvement to the code base,
 please don't hesitate to contact us at jf3423@columbia.edu or submit a pull request 😄