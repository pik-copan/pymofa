
# coding: utf-8

# # pymofa tutorial 1
# 
# ## Introduction
# 
# This notebook introduces the basic functionalities of pymofa, the python modeling framework to run and evaluate your models systematically
# 
# The last update happed on:

# In[1]:

import datetime
print(datetime.datetime.now().date())


# It does not hurt to locally install pymofa on your system by executing
# 
#     &pip install -e .
#     
# at the pymofa root folder.
# 
# If you prefer not to and want to work with this notebook interactively, exectue

# In[2]:

# cd ..


# to be at the pymofa root.
# 
# Can be left out if `pymofa` is (locally) installed at your system.

# In[3]:

import numpy as np
import pandas as pd


# ## A discrete predetor prey dummy model
# First we need to create a dummy model. Let's use a discrete version of the famous predator prey model.
# 

# In[4]:

def predprey_model(prey_birth_rate, prey_mortality, 
                   predator_efficiency, predator_death_rate,
                   initial_prey, initial_predators,
                   time_length):
    """Discrete predetor prey model."""
    A = -1 * np.ones(time_length)
    B = -1 * np.ones(time_length)
    A[0] = initial_prey
    B[0] = initial_predators
    for t in range(1, time_length):
        A[t] = A[t-1] + prey_birth_rate * A[t-1] - prey_mortality * B[t-1]*A[t-1]
        B[t] = B[t-1] + predator_efficiency * B[t-1]*A[t-1] - predator_death_rate * B[t-1] +            0.02 * (0.5 - np.random.rand())
    return A, B

# ## Applying pymofa

# In[7]:

# imports
from pymofa.experiment_handling import experiment_handling as eh
import itertools as it


# In[8]:

# Path where to Store the simulated Data
SAVE_PATH_RAW = "./dummy/pmX01data"


# In[9]:

# Definingh the experiment execution function
#      it gets paramater you want to investigate, plus `filename` as the last parameter
def RUN_FUNC(dummyP, prey_birth_rate,
             coupling,
             predator_death_rate,
             initial_pop,
             time_length):  
    """Insightful docstring."""
    # poss. process
    prey_mortality = coupling
    predator_efficiency = coupling
    initial_prey = initial_pop
    initial_predators = initial_pop
    # one could also do more complicated stuff here, e.g. drawing something from a random distribution
    
    # running the model
    preys, predators = predprey_model(prey_birth_rate,
                                      prey_mortality,
                                      predator_efficiency,
                                      predator_death_rate,
                                      initial_prey,
                                      initial_predators,
                                      time_length)
    
    # preparing the data
    res = pd.DataFrame({"preys": np.array(preys),
                        "predators": np.array(predators)})
    res.index.name = "tstep"
    
    # store run funcs model result
    # store(res)
    
    # determine exit status (if something went wrong)
    # if exit status > 0 == run passed
    # if exit status < 0 == Run Failed
    exit_status = 42
    
    # RUN_FUNC needs to return exit_status 
    return exit_status, res


# NOTE: runfunc result dataframe columns need to be in the same order
# Better to give them alphabetically

# In[10]:

RUNFUNC_RESULTSFORM = pd.DataFrame(columns=["predators", "preys"])
RUNFUNC_RESULTSFORM.index.name = "tstep"


# **IMPORTANT NOTE
# DO NOT USE `np.arrange` to specify parameter ranges
# ** causes rounding problems

# In[11]:

# Parameter combinations to investiage
dummyPs = ["A", "B"]
prey_birth_rate = [0.1, 0.2]
predator_death_rate = [0.01]
coupling = [0.1]

initial_pop = [1.0, 1.1]
time_length = [1000]

PARAM_COMBS = list(it.product(dummyPs,
                              prey_birth_rate,
                              coupling,
                              predator_death_rate,
                              initial_pop,
                              time_length))


# In[12]:

# Sample Size
SAMPLE_SIZE = 2


# NOTE: Index should be internally generated

# In[13]:

# initiate handle instance with experiment variables
handle = eh(RUN_FUNC,
            RUNFUNC_RESULTSFORM,
            PARAM_COMBS,
            SAMPLE_SIZE,
            SAVE_PATH_RAW)

if __name__ == "__main__":
    handle.compute()

# In[14]:

# ## Obtaining the saved data
# by queriny the hdf5 store
# 
# see http://pandas.pydata.org/pandas-docs/stable/io.html#querying for details
# 


# ## pymofa again
# 
# suppose you want to increase the sample size:

# In[18]:

SAMPLE_SIZE = 2
prey_birth_rate = [0.1]
predator_death_rate = [0.01, 0.02]
#initial_pop = [1.1, 1.0, 2]
PARAM_COMBS = list(it.product(dummyPs, prey_birth_rate,
                              coupling, predator_death_rate,
                              initial_pop, time_length))

# You can now re instanciate the experiment handle and compute the experiment all over agian. Expect that pymofa detacts that is has already computed something and wont to that again.

# In[19]:

# initiate handle instance with experiment variables<

handle = eh(RUN_FUNC,
            RUNFUNC_RESULTSFORM,
            PARAM_COMBS,
            SAMPLE_SIZE,
            SAVE_PATH_RAW)


# In[20]:

# Compute experiemnts raw data
if __name__ == "__main__":
    handle.compute()
    

# ## Postprocessing or resaving
# Supoose we want to  analyse our experiment data by computing some statistics on it. We can do is as well with pymofa.

# Let's say that we are intersted in the sum of preys and predators in the first 250 time steps.

# In[180]:



