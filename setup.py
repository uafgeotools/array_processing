from setuptools import setup, find_packages

config = {'name':               'array_processing',
          'url':                'https://github.com/uafgeotools/array_processing',
          'packages':           find_packages(),
          'install_requires':   'waveform_collection'
         }

setup(**config)
