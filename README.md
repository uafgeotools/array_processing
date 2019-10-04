array_processing
================

Various array processing tools for infrasound and seismic data. By default uses
weighted least-squares to determine the trace velocity and back-azimuth of a
plane wave crossing an array. More advanced processing (such as least-trimmed
squares) is easily integrated.

Installation
------------

It is recommended to install this package into a new or pre-existing
environment, such as that provided by 
[conda](https://docs.conda.io/projects/conda/en/latest/index.html).
(Ensure that the environment contains all of
the packages listed in the [Dependencies](#dependencies) section.)

To create a new conda environment for use with this and other _uafgeotools_
packages, execute the following terminal command:
```
$ conda create --name uafinfra --channel conda-forge obspy
```
This creates a new environment called `uafinfra` with ObsPy and its dependencies
installed.

Next, install _waveform_collection_. Execute the following terminal commands:
```
$ conda activate uafinfra
$ git clone https://github.com/uafgeotools/waveform_collection.git
$ cd waveform_collection
$ pip install --editable .
```
The final command installs the package in "editable" mode, which enables the 
installed package to be updated with a `git pull` in the local repository. This
install command only needs to be run once.

Finally, install _array_processing_ in a similar manner to _waveform_collection_.
```
$ conda activate uafinfra
$ git clone https://github.com/uafgeotools/array_processing.git
$ cd array_processing
$ pip install --editable .
```

Dependencies
------------

_uafgeotools_ packages:

* [_waveform_collection_](https://github.com/uafgeotools/waveform_collection)

Python packages:

* [ObsPy](http://docs.obspy.org/)


Usage
-----

Import the package like any other python package, ensuring the correct environment
is active. For example,
```
$ conda activate uafinfra
$ python
>>> import array_processing
```
Currently, documentation only exists in function docstrings. For a
usage example, see [`example.py`](example.py).


Authors
-------

(_Alphabetical order by last name._)

Jordan Bishop  
David Fee  
Curt Szuberla
Andrew Winkelman


**References**

Least squares and array uncertainty:

Szuberla, C. A. L., & Olson, J. V. (2004). Uncertainties associated with
parameter estimation in atmospheric infrasound arrays. J. Acoust. Soc. Am.,
115(1), 253–258. https://doi.org/doi:10.1121/1.1635407

Least-trimmed squares:

Bishop, J.W., Fee, D., & Szuberla, C. A. L., 2019. Improved infrasound array
processing with robust estimators, Geophysical Journal International, p. In prep.
