

"""
Run a computer model for various parameter combinations and sample sizes.

This "experiment_handling" (eh) module provides functionality to run a computer
model for various parameter combinations and sample sizes.
@author: wbarfuss

implementation of native mpi4py dependency and
post processing of pandas dataframes by jakobkolb

mpi4py implementation heavily relies on
http://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py

TODOs
-----
bad runs are retried immediately. needs a stop method though in
case some parameter combinations are impossible to complete.

resaving is parallelized such that parameter combinations are
distributed amongs nodes. For small sample sizes, a serial
implementation could be faster due to overhead...
"""
from __future__ import print_function

import glob
import os
import sys
import traceback
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from mpi4py import MPI


def enum(*sequential, **named):
    """
    Handy way to fake an enumerated type in python.

    Example
    -------
    >>> Numbers = enum(ONE=1, TWO=2, THREE='three')
    >>> Numbers.ONE
    1
    >>> Numbers.TWO
    2
    >>> Numbers.THREE
    'three'

    Source
    ------
    "http://stackoverflow.com/questions/36932/
     how-can-i-represent-an-enum-in-python"
    """
    enums = dict(list(zip(sequential, list(range(len(sequential))))), **named)
    return type('Enum', (), enums)


tags = enum('START', 'READY', 'DONE', 'FAILED', 'EXIT')


class experiment_handling(object):
    """Class doc string."""

    def __init__(self, sample_size, parameter_combinations, index, path_raw,
                 path_res='./data/', use_kwargs=False):
        """
        Set up the experiment handling class.

        saves the sample size and parameter combinations of the
        experiment and creates a list of tasks containing parameter
        combinations, ensemble index and path to the corresponding
        output files.

        also sets up the MPI environment to keep track of
        master and subordinate nodes.

        Parameters
        ----------
        sample_size : int
            number of runs for each parameter combination e.g.
            size of the ensemble for statistical evaluation
        parameter_combinations: list[tuples]
            list of parameter combinations that are stored in
            tuples. Number of Parameters for each combination
            has to fit the number of input Parameters of the
            run function.
        index : dict
            indicating the varied Parameters as
            {position in parameter_combinations: name}
            is needed for post processing to create the
            multi index containing the variable Parameters
        path_raw : string
            absolute path to the raw data of the computations
        path_res : string
            absolute path to the post processed data

        """

        print('initializing pymofa experiment handle')

        self.sample_size = sample_size

        self.kwparameter_combinations = \
            [{index[i]: params[i]
              for i in index.keys()}
             for params in parameter_combinations]

        self.parameter_combinations = parameter_combinations
        self.use_kwargs = use_kwargs
        self.index = index
        self.index_names = [self.index[key] for key in range(len(self.index))]

        # add "/" to paths if missing
        self.path_raw = path_raw + "/" if not path_raw.endswith("/") else \
            path_raw
        self.path_res = path_res + "/" if not path_res.endswith("/") else \
            path_res

        # load mpi4py MPI environment and get size and ranks
        self.comm = MPI.COMM_WORLD
        self.status = MPI.Status()
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # split size of environment in 1 master and n-1 slaves
        self.master = 0
        self.nodes = list(range(1, self.size))
        self.n_nodes = self.size - 1

        # tell process whether it is master or slave
        if self.rank == 0:
            self.amMaster = True
            self.amNode = False
            print('detected {} nodes in MPI environment'.format(self.size))
        else:
            self.amMaster = False
            self.amNode = True

        # create list of tasks (parameter_combs*sample_size)
        # and paths to save the results.
        self.tasks = []

        for s in range(self.sample_size):
            for p, kwp in zip(self.parameter_combinations,
                              self.kwparameter_combinations):
                filename = self.path_raw + self._get_id(p, s)
                if not os.path.exists(filename):
                    if self.use_kwargs:
                        self.tasks.append((kwp, filename))
                    else:
                        self.tasks.append((p, filename))

        self.filenames = []

    def compute(self, run_func, skipbadruns=False):
        """
        Compute the experiment.

        Call the 'run_func' for several 'parameter_combiantions' and
        'sample_size's and provides a unique ID for each run to store the data
        at the 'path_res' + ID

        Parameters
        ----------
        run_func : function
            The function the executes the Model for a given set of Parameters.
            The first P paramters need to fit to the parameter_combinations.
            The last parameter of run_func has to be named filename
            If run_func succeded, it returns >=0, else it returns < 0
        skipbadruns : bool (Default: False)
            If you don't want this function to check for bad runs that shall be
            recalculated, than set to "True". Possible reason: speed.

        """
        assert (callable(run_func)), "run_func must be callable"
        assert (isinstance(skipbadruns, bool)), "scipbadruns must be boolean"
        if self.amMaster:
            # check, if path exists. If not, create.
            if not os.path.exists(self.path_raw):
                os.makedirs(self.path_raw)

            # give brief feedback about remaining work.
            print(str(len(self.tasks)) + " of "
                  + str(len(self.parameter_combinations) * self.sample_size)
                  + " single computations left")

            # check if nodes are available. If not, do serial calculation.
            if self.n_nodes < 1:
                print("Only one node available. No parallel execution.")

                for task in self.tasks:
                    (params, filename) = task
                    result = -1
                    while result < 0:
                        if self.use_kwargs:
                            result = run_func(filename=filename, **params)
                        else:
                            result = run_func(*params, filename=filename)

            print("Splitting calculations to {} nodes.".format(self.n_nodes))
            sys.stdout.flush()

            task_index = 0
            tasks_completed = 0
            closed_nodes = 0
            while closed_nodes < self.n_nodes:
                n_return = self.comm.recv(source=MPI.ANY_SOURCE,
                                          tag=MPI.ANY_TAG, status=self.status)
                source = self.status.Get_source()
                tag = self.status.Get_tag()
                if tag == tags.READY:
                    # node is ready, can take new task
                    if task_index < len(self.tasks):
                        self.comm.send(self.tasks[task_index], dest=source,
                                       tag=tags.START)
                        task_index += 1
                    else:
                        self.comm.send(None, dest=source, tag=tags.EXIT)
                elif tag == tags.FAILED:
                    # node failed to complete.
                    # retry (failed runs send their task as return)
                    self.comm.send(n_return, dest=source, tag=tags.START)
                elif tag == tags.DONE:
                    # node succeeded
                    tasks_completed += 1
                    self._progress_report(tasks_completed, len(self.tasks),
                                          "Calculating...")
                elif tag == tags.EXIT:
                    # node completed all tasks. close
                    closed_nodes += 1
            print("Calculating 0 ...done.")

        if self.amNode:
            # Nodes work as follows:
            # name = MPI.Get_processor_name() <-- unused variable
            while True:
                self.comm.send(None, dest=self.master, tag=tags.READY)
                task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG,
                                      status=self.status)
                tag = self.status.Get_tag()

                if tag == tags.START:
                    # go work:
                    (params, filename) = task
                    if self.use_kwargs:
                        result = run_func(filename=filename, **params)
                    else:
                        result = run_func(*params, filename=filename)
                    if result >= 0:
                        self.comm.send(result, dest=self.master, tag=tags.DONE)
                    else:
                        self.comm.send(task, dest=self.master, tag=tags.FAILED)
                elif tag == tags.EXIT:
                    break

            self.comm.send(None, dest=self.master, tag=tags.EXIT)

        self.comm.Barrier()

    def resave(self, eva, name, no_output=False):
        """
        Postprocess the computed raw data.

        Using the operators that are provided in the eva dictionary.

        Supports trajectories as data for parameter combinations if they are
        saved as data frames. Data frames must have a one dimensional index
        with the time stamps of the measurements as index values.
        Time stamps must be consistent between trajectories.

        Parameters
        ----------
        eva : dict as {<name of macro-quantities> : function how to compute it}
            The function must receive a list of filenames
        name : string
            The name of the saved macro quantity pickle file
        no_output: bool
            tell resave that eva will not yield any output.

        """
        # ALL NODES NEED TO KNOW WHETHER EVA RETURNS A DATAFRAME.

        # First, prepare list of effective parameter combinations for MultiIndex
        if self.use_kwargs:
            eff_params = {self.index[k]:
                          np.unique([p[self.index[k]] for p in self.kwparameter_combinations])
                          for k in self.index.keys()}
        else:
            eff_params = {self.index[k]:
                          np.unique([p[k] for p in self.parameter_combinations])
                          for k in self.index.keys()}

        # if eva returns a data frame,
        # add the indices and column names to the list of effective parameters.

        if no_output:
            process_df = False
        else:
            # Therefore, first get the filenames for the first parameter combination that has output files
            i = 0
            while True:
                filenames_p0 = np.sort(glob.glob(self.path_raw + self._get_id(self.parameter_combinations[i])))
                i += 1
                if len(filenames_p0) > 0:
                    break

            # and get the eva returns for the first callable for these filenames
            eva_return = self._evaluate_eva(eva,
                                            list(eva.keys())[0],
                                            filenames_p0,
                                            'building index with parameters {}'.format(self.parameter_combinations[-1]))

            # if the eva returns a dataframe, add names to eff_params
            if isinstance(eva_return, pd.core.frame.DataFrame) and \
                    not isinstance(eva_return.index, pd.MultiIndex):

                eff_params['timesteps'] = eva_return.index.values
                eff_params['observables'] = eva_return.columns.values

                process_df = True

            # else, do nothing, but note, that return is NOT a dataframe
            else:
                process_df = False

        if self.amMaster:
            print('processing: ', name)

            # create save_path if it is not yet existing
            if not os.path.exists(self.path_res):
                os.makedirs(self.path_res)

            # Create empty MultiIndex and Dataframe

            n_index_levels = len(self.index_names)
            m_index = pd.MultiIndex(levels=[[]]*n_index_levels,
                                    codes=[[]]*n_index_levels,
                                    names=self.index_names)

            df = pd.DataFrame(index=m_index)

            # initialize counters for work sharing amongst nodes
            task_index = 0
            tasks_completed = 0
            n_tasks = len(self.parameter_combinations) * len(list(eva.keys()))
            closed_nodes = 0

            # Check if nodes are available. If not, do serial computation.
            if self.n_nodes < 1:
                print("Only one node available. No parallel execution.")
                for task_index in range(n_tasks):
                    p_index, k_index = divmod(task_index, len(list(eva.keys())))
                    p, key = (self.parameter_combinations[p_index], list(eva.keys())[k_index])
                    fnames = np.sort(glob.glob(self.path_raw + self._get_id(p)))

                    eva_return = self._process_eva_output(eva=eva,
                                                          key=key,
                                                          p=p,
                                                          fnames=fnames,
                                                          process_df=process_df)
                    if not no_output:
                        df = df.append(other=eva_return, verify_integrity=True)

            # If nodes are available, distribute work amongst nodes.

            while closed_nodes < self.n_nodes:
                # master keeps subordinate nodes buzzy:
                data = self.comm.recv(source=MPI.ANY_SOURCE,
                                      tag=MPI.ANY_TAG,
                                      status=self.status)
                source = self.status.Get_source()
                tag = self.status.Get_tag()
                if tag == tags.READY:
                    # node ready to work.
                    if task_index < n_tasks:
                        # if there is work, distribute it
                        p_index, k_index = divmod(task_index, len(list(eva.keys())))
                        task = (self.parameter_combinations[p_index], list(eva.keys())[k_index])
                        self.comm.send(task, dest=source, tag=tags.START)
                        task_index += 1
                    else:
                        # if not, release worker
                        self.comm.send(None, dest=source, tag=tags.EXIT)
                elif tag == tags.DONE:
                    (mx, key, eva_return) = data
                    if not no_output:
                        df = df.append(eva_return)
                    tasks_completed += 1
                    self._progress_report(tasks_completed, n_tasks,
                                          "Post-processing...")
                elif tag == tags.EXIT:
                    closed_nodes += 1
            if not no_output:
                df = df.unstack(level='key')
                df.columns = df.columns.droplevel()
                df.to_pickle(self.path_res + name)
            print('\nDone')

        if self.amNode:
            # Nodes work as follows:
            while True:
                self.comm.send(None, dest=self.master, tag=tags.READY)
                task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG, status=self.status)
                tag = self.status.Get_tag()
                if tag == tags.START:
                    # go work:
                    (p, key) = task
                    mx = tuple(p[k] for k in list(self.index.keys()))
                    fnames = np.sort(glob.glob(self.path_raw
                                               + self._get_id(p)))

                    eva_return = self._process_eva_output(eva=eva,
                                                          key=key,
                                                          p=p,
                                                          fnames=fnames,
                                                          process_df=process_df)

                    self.comm.send((mx, key, eva_return),
                                   dest=self.master,
                                   tag=tags.DONE)
                elif tag == tags.EXIT:
                    break

            self.comm.send(None, dest=self.master, tag=tags.EXIT)

        self.comm.Barrier()

#    @staticmethod
    def _evaluate_eva(self, eva, key, fnames, msg=None):
        """Evaluate eva for given key and filenames.

        Also and do proper error logging to enable debugging.

        Parameters
        ----------
        eva: dict
            dictionary of callables,
        key: key in eva dictionary
            key for the callable in the eva dictionary, that is
            to be evaluated,
        fnames: list of strings
            names of files to evaluate callable in eva
            dictionary on.
        msg: basestring
            some message to print to track where the
            method has been called.

        Returns
        -------
        eva_return: the return value of the evaluated callable

        """
        try:
            eva_return = eva[key](fnames)
        except ValueError:
            print('value error in eva of {} at {}\n'.format(key, msg))
            traceback.print_tb(ValueError)
            eva_return = None

        return eva_return

    def _process_eva_output(self, eva,
                            p, key, fnames, process_df):
        """Process the output of the callable eva[key].

        Yield a pandas data frame that can be appended to the final data structure.

        Parameters
        ----------
        eva: dict as {<name of macro-quantities> : function how to compute it}
             The function must receive a list of file names,
        p: tuple
            values for variables that are to include in the
            resulting data frames index,
        key: key for eva dict
        fnames: list
            list of file names to interpret by eva,
        process_df: bool
            indicator whether the return of eva is a data frame.

        Returns
        -------
        eva_return: pandas Dataframe

        """
        eva_return = self._evaluate_eva(eva, key, fnames, 'processing data for parameters {} with {} files'.format(p, len(fnames)))

        if eva_return is None:
            return

        if process_df:
            index_names = self.index_names + ["timesteps", "observables"]
            # create data frame with additional levels in index
            eva_return = eva_return.stack(level=0)
            # therefore, first create the multi index.
            # 1) find length of codes
            codes_length = len(eva_return.index.codes[0])
            # 2) add new levels to codes (being zero, since new
            # index levels have constant values
            index_codes = [[0] * codes_length] \
                * (len(self.index.keys()) + 1) \
                + eva_return.index.codes
            # 3) add new index levels to the old ones
            index_levels = [[key]] + [[list(p)[l]]
                                      for l in self.index.keys()] \
                + eva_return.index.levels
            # 4) and fill it all into the multi index
            m_index = pd.MultiIndex(levels=index_levels,
                                    codes=index_codes,
                                    names=['key'] + list(index_names))
            # then create the data frame
            return pd.DataFrame(index=m_index,
                                data=eva_return.values)
        elif not process_df:
            index_names = self.index_names
            # same as above but without levels and codes from eva return
            codes_length = 1
            index_codes = [[0] * codes_length] * (len(self.index.keys()) + 1)
            index_levels = [[key]] + [[list(p)[l]] for l in self.index.keys()]
            tmp_index_names = ['key'] + list(index_names)
            m_index = pd.MultiIndex(levels=index_levels,
                                    codes=index_codes,
                                    names=tmp_index_names)
            return pd.DataFrame(index=m_index,
                                data=[eva_return])

    @staticmethod
    def _get_id(parameter_combination, i=None):
        """
        Get a unique ID for a `parameter_combination` and ensemble index `i`.

        ID is of the form 'parameterID_index_ID.pkl'

        Parameters
        ----------
        parameter_combination : tuple
            The combination of Parameters
        i : int
            The ensemble index.
            if i = None, it returns the pattern
            matching the entire ensemble e.g.
            parameterID*.pkl

        Returns
        -------
        ID : string
            unique ID or pattern plus the ".pkl" ending

        """
        res = str(parameter_combination)  # convert to sting
        res = res[1:-1]  # delete brackets
        res = res.replace(", ", "-")  # remove ", " with "-"
        res = res.replace(",", "-")  # remove "," with "-"
        res = res.replace(".", "o")  # replace dots with an "o"
        res = res.replace("'", "")  # remove 's from values of string variables
        # Remove all the other left over mean
        # charakters that might fuck with you
        # bash scripting or wild card usage.
        for mean_character in "[]()^ #%&!@:+={}'~":
            res = res.replace(mean_character, "")
        if i is None:
            res += '*.pkl'
        else:
            res += "_s" + str(i)  # add sample size
            res += ".pkl"  # add file type

        return res

    @staticmethod
    def _progress_report(i, loop_length, msg=""):
        """Print a small progress report for a loop of defined length.

        Parameters
        ----------
        i : int
            the current position in the loop
        loop_length : int
            the length of the loop
        msg : string
            (optional) a preceding string

        """
        sys.stdout.write("\r")
        sys.stdout.flush()
        sys.stdout.write("{} {:.2%}".format(msg, float(i)/float(loop_length)))
        sys.stdout.flush()

        if i == loop_length - 1:
            sys.stdout.write("\n")
            sys.stdout.flush()


def even_time_series_spacing(dfi, n, t0=None, t_n=None):
    """Interpolate irregularly spaced time series.

    To obtain regularly spaced data.

    Parameters
    ----------
    dfi : pandas dataframe
        dataframe with one dimensional index containing
        time stamps as index values.
    n   : int
        number of regularly spaced timestamps in the
        resulting time series.
    t0  : float
        starting time of the resulting time series -
        defaults to the first time step of the input
        time series.
    t_n  : float
        end time of the resulting time series -
        defaults to the last time step of the input
        time series.

    Returns
    -------
    dfo : pandas dataframe
        pandas dataframe with n regularly spaced
        time steps starting at t0 and ending at t_n
        inclusively. Output data is interpolated from
        the input data.

    """
    if t0 is None:
        t0 = dfi.index.values[0]
    if t_n is None:
        t_n = dfi.index.values[-1]

    timestamps = np.linspace(t0, t_n, n)

    observables = dfi.columns
    measurements = {}

    # assuming the time series just breaks at some point and
    # continues with Nan's we want to extract the intact part.

    i_max = sum(~np.isnan(dfi[observables[0]]))
    for o in observables:
        measurements[o] = list(dfi[o])[:i_max]
    measurements['time'] = list(dfi.index.values)[:i_max]
    # t_min = measurements['time'][0] <-- unused variable
    t_max = measurements['time'][-1]

    # create interpolation functions for intact part of
    # time series:

    interpolations = {}
    for o in observables:
        interpolations[o] = interp1d(measurements['time'], measurements[o])

    # generate data points from interpolation functions
    # fill series with Nan's from where the original time
    # series ended.

    data_points = {'time': timestamps}
    for o in observables:
        x = [t if t < t_max else float('NaN') for t in timestamps]
        data_points[o] = interpolations[o](x)

    dfo = pd.DataFrame(data_points, index=timestamps)

    return dfo


if __name__ == '__main__':
    eh = experiment_handling(sample_size=10,
                             parameter_combinations=[1, 2],
                             index={'phi': 1},
                             path_raw='./')
