from setuptools import setup, find_packages
import os

# https://github.com/readthedocs/readthedocs.org/issues/5512#issuecomment-475073310
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['waveform_collection', 'lts_array']

config = {'name':             'array_processing',
          'url':              'https://github.com/uafgeotools/array_processing',
          'packages':         find_packages(),
          'install_requires': ['waveform_collection', 'lts_array']
          }

setup(
      name='array_processing',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES
      )
