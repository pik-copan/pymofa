"""setup.py defining some installation instructions."""
from setuptools import setup

setup(name='pymofa',
      version='0.1',
      description='experimentation environment for mpi powered ensemble' +
      'runs on high performance cluster',
      url='http://githum.com/wbarfuss/pymofa',
      author='Wolfram Barfuss',
      author_email='barfuss@pik-potsdam.de',
      license='MIT',
      packages=['pymofa'],
      zip_safe=False)
