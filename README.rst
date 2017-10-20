
python modeling framework (pyMoFa)
==================================
is a collection of simple functions to run and evaluate computer models
systematically.

.. contents::

Disclaimer
----------
This is free software - use at your own risk and convenience.


Usecase
-------
* You have some sort of computer model you want to do parameter studies with

Features
--------
* Computes parallel
* Works iteratively - *pymofa checks wheter you have already computed some task
  and won't compute these again*

Design
------
* With pymofa you write one python file to set up one computer experiment
* This python file will contain a function (called the RUN_FUNC) that configures and exectues your model run
* The parameters of the RUN_FUNC will be the parameters of the experiment

This means:
* Raw data will be stored with these parameters
* You will need to give pymofa a list of parameter combination (i.e. a tuple
  of parameter values of the same length as the parameters)
* If you want to change the parameters, write a new experiment 

Usage
-----
Please have a look at the tutorials (either interactivly after downloading this
repository or by starting `here <https://github.com/wbarfuss/pymofa/blob/master/tutorial/01_RunningAModel.ipynb>`_)

For further documentation, use the source!

Tests
-----
using `pytest <http://docs.pytest.org/en/latest/>`_ with
`pylama <https://github.com/klen/pylama#pytest-integration>`_
(including `pylama-pylint <https://github.com/klen/pylama_pylint>`_)
and test coverage reports with the `pytest` plugin
`pytest-cov <https://github.com/pytest-dev/pytest-cov>`_.

To be installed with::

    $> pip install pytest pylama pylama-pylint pytest-cov
    
The config file is <pytest.ini>.
    
Write tests and make sure that they pass by::

    $> py.test

