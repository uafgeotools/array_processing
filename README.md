array_processing
================

[![](https://readthedocs.org/projects/uaf-array-processing/badge/?version=master)](https://uaf-array-processing.readthedocs.io/)

Various array processing tools for infrasound and seismic data. By default uses
least-squares to determine the trace velocity and back-azimuth of a plane wave
crossing an array in sliding time windows. More advanced processing (such as
least-trimmed squares) is easily integrated. Also provides tools to characterize
the array response, uncertainty, source-location of a spherical wave crossing
the array, etc. See
[documentation](https://uaf-array-processing.readthedocs.io/) and
[`example.py`](https://github.com/uafgeotools/array_processing/blob/master/example.py)
for more info.

**General References and Suggested Citations**

_Least squares and array uncertainty:_

Szuberla, C. A. L., & Olson, J. V. (2004). Uncertainties associated with
parameter estimation in atmospheric infrasound arrays, J. Acoust. Soc. Am.,
115(1), 253–258.
[https://doi.org/doi:10.1121/1.1635407](https://doi.org/doi:10.1121/1.1635407)

_Least-trimmed squares:_

Bishop, J. W., Fee, D., & Szuberla, C. A. L. (2020). Improved infrasound array
processing with robust estimators, Geophys. J. Int., 221 p. 2058-2074.
[https://doi.org/10.1093/gji/ggaa110](https://doi.org/10.1093/gji/ggaa110)

Installation
------------

We recommend you install this package into a new
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
(Please install [Anaconda](https://www.anaconda.com/products/individual) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.)
The environment must contain all of the packages listed in the
[Dependencies](#dependencies) section. For ease of installation, we've provided
an
[`environment.yml`](https://github.com/uafgeotools/array_processing/blob/master/environment.yml)
file which specifies all of these dependencies as well as instructions for
installing _array_processing_ itself. To install _array_processing_ in this
manner, execute the following commands:
```
git clone https://github.com/uafgeotools/array_processing.git
cd array_processing
conda env create -f environment.yml
```
This creates a new conda environment named `uafinfra` and installs
_array_processing_ and all of its dependencies there.

The final command in the `environment.yml` file installs _array_processing_ in "editable" mode, which means that you can update it with a simple `git pull` in your local repository. We recommend you do this often, since this code is still under rapid development.

<details>
<summary>
For installation into a pre-existing conda environment, click here.
</summary>
<br>
First ensure you have ObsPy and FastKML installed (<code>conda install -c conda-forge
obspy fastkml</code>) and then download and install the <em>uafgeotools</em>
dependencies and this package with:
<br>
<br>

```
pip install git+https://github.com/uafgeotools/waveform_collection.git
pip install git+https://github.com/uafgeotools/lts_array.git
pip install git+https://github.com/uafgeotools/array_processing.git
```
(Note that this option does not produce a local clone of the repository.)
</details>

Dependencies
------------

_uafgeotools_ packages:

* [_waveform_collection_](https://github.com/uafgeotools/waveform_collection)
* [_lts_array_](https://github.com/uafgeotools/lts_array)

Python packages:

* [ObsPy](https://docs.obspy.org/)
* [FastKML](https://fastkml.readthedocs.io/)

Usage
-----

Import the package like any other Python package, ensuring the correct
environment is active. For example,
```
$ conda activate uafinfra
$ python
>>> import array_processing
```
Documentation is available online
[here](https://uaf-array-processing.readthedocs.io/). For a usage example, see
[`example.py`](https://github.com/uafgeotools/array_processing/blob/master/example.py).

Note: The `array_plot()` function does not allow both `sigma_tau` and dropped elements from least trimmed squares to be plotted.

The `sigma_tau` variable is an indicator of nonplanar propagation across an array (using all elements), and least trimmed squares drops element pairs that appear "too far" from planar. In this way, having a large `sigma_tau` value and having consistently dropped element pairs (while not the same) suggest a departure from the plane wave model.

`sigma_tau` is only calculated when ordinary least squares (`ALPHA=1.0`) is used. The ability to plot one or the other was intended as a safeguard against potentially conflicting processing assumptions. To maintain a consistent output data structure, the `sigma_tau` key returns a `np.nan` in the case that subset pairs are trimmed (0.5 <= ALPHA < 1.0).

If `ALPHA = 1.0`, the dropped stations are not plotted since least trimmed squares is not used, and `sigma_tau` may be plotted if specified. If ALPHA < 1.0, then `sigma_tau` is not plotted or calculated.


Authors
-------

(_Alphabetical order by last name._)

Jordan Bishop<br>
David Fee<br>
Curt Szuberla<br>
Liam Toney<br>
Andrew Winkelman
