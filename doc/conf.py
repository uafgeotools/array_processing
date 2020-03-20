import os
import sys

sys.path.insert(0, os.path.abspath('../array_processing'))

project = 'array_processing'

html_show_copyright = False

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'recommonmark',
    'sphinx.ext.viewcode',
    'sphinxcontrib.apidoc',
    'sphinx.ext.mathjax',
]

html_theme = 'sphinx_rtd_theme'

napoleon_numpy_docstring = False

master_doc = 'index'

autodoc_mock_imports = ['numpy',
                        'obspy',
                        'scipy'
                        ]

apidoc_module_dir = '../array_processing'

apidoc_output_dir = 'api'

apidoc_separate_modules = True

apidoc_toc_file = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'obspy': ('https://docs.obspy.org/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'waveform_collection': ('https://uaf-waveform-collection.readthedocs.io/en/master/', None),
}
