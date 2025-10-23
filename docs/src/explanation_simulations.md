### Spike generation

After summing the generator potential from the different temporal models, we apply firing rate gain (spikes/(second * unit contrast)), parameter *A* in the [Victor 1987](references.md#victor-1987) model, values from [Benardete and Kaplan 1999](references.md#benardete-1999) for parasol units and [Benardete and Kaplan 1997](references.md#benardete-1997) for midget units. This gives us a distribution of gains. Next this value is calibrated across the different temporal and spatial model combinations to give a threshold firing rate for 3.5% contrast (parasol units) or 11.4% contrast (midget units). We run separately luminance contrast calibration (following [Lee et al. 1989](references.md#lee-1989)) and drifting grating calibration (following [Derrington and Lennie 1984](references.md#derrington-1984)).

Next the signal goes through a threshold linear (rectified linear, ReLu) nonlinearity.

Finally the spike generation comes from either a `poisson` process or from a `refractory` model. For refractory model the recovery function is from [Berry and Meister 1998](references.md#berry-1998) and absolute and relative refractory parameters were estimated from [Uzzell and Chichilnisky 2004](references.md#uzzell-2004), Fig 7B, bottom row, inset. Currently we have only one set of fixed refracory model parameters, shared across all GC units.


### Noise model

Spontaneous firing rates in midgets and parasol units ranges from 5 to over 30 Hz ([Appleby and Manookin 2019](references.md#appleby-2019), [Sinha et al. 2017](references.md#sinha-2017)); we fixed the cone noise to induce ~25Hz baseline firing rates for the ON parasol and midget units, and ~5Hz for the OFF parasol and midget units (Raunak Sinha, personal communication). 

The noise type `shared` gets cone spectrum from [Angueyra and Rieke 2013](references.md#angueyra-2013), and this drives the ganglion cell background firing rates. You can use the same cone noise for all ganglion cell types, hypothetically resulting in shared downstream signal for a shared stimulus. The noise type `independent` fits Gaussian distribution to the GC noise firing rates from the shared noise and next samples indipendent noise for each GC unit from this distribution.
