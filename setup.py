"""Setup.py of pymofa."""

from setuptools import setup

# for developers: recommended way of installing is to run in this directory
# pip install -e .
# This creates a link insteaed of copying the files,
# so modifications in this directory are
# modifications in the installed package.

setup(name="pymofa",
      version="0.1.0",
      description="PYthon MOdelling FrAmework",
      url="https://github.com/wbarfuss/pymofa",
      author="Copan-group @ PIK",
      author_email="barfuss@pik-potsdam.de",
      license="MIT",
      packages=["pymofa"],
      install_requires=[
            "numpy>=1.11.0",
            "scipy>=0.17.0",
            "sympy>=1.0",
            "mpi4py>=2.0.0",
            "pandas>=0.19.0",
      ],
      # see http://stackoverflow.com/questions/15869473/
      # what-is-the-advantage-of-setting-zip-safe-to-true-when-packaging-a
      # -python-projec
      zip_safe=False
      )
