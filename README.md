array_processing
===============

Various array processing tools for infrasound and seismic data. By default uses
weighted least-squares to determine the trace velocity and back-azimuth of a 
plane wave crossing an array. More advanced processing (such as least-trimmed
squares) are easily integrated.


**References**

Least squares and array uncertainty:

Szuberla, C. A. L., & Olson, J. V. (2004). Uncertainties associated with parameter estimation in atmospheric infrasound arrays. J. Acoust. Soc. Am., 115(1), 253–258. https://doi.org/doi:10.1121/1.1635407

Least-trimmed squares:

Bishop, J.W., Fee, D., & Szuberla, C. A. L., 2019. Improved infrasound array processing with robust estimators, Geophysical Journal International, p. In prep.


Dependencies
------------

_uafgeotools_ repositories:

* [_waveform_collection_](https://github.com/uafgeotools/waveform_collection)

* [Python](https://www.python.org/) >= 3.2

Python packages:

* [ObsPy](http://docs.obspy.org/)

...and its dependencies, which you don't really have to be concerned about if
you're using [conda](https://docs.conda.io/projects/conda/en/latest/index.html)!

It's recommended that you create a new conda environment to use with this
repository:
```
conda create -n array_processing -c conda-forge obspy "python>=3.2"
```

Usage
-----

To use _array_processing_, clone or download this repository and any additional
_uafgeotools_ repository dependencies and add them to your `PYTHONPATH`, e.g.
in a script where you'd like to use _rtm_:
```python
import sys
sys.path.append('/path/to/waveform_collection')
sys.path.append('/path/to/array_processing')
```
Then you can access package functions with (for example)
```python
from waveform_collection import gather_waveforms
from array_processing import array_tools
```
and so on.

Authors
-------

(_Alphabetical order by last name._)

Jordan Bishop
David Fee
Curt Szuberla  