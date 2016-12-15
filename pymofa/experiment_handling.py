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
import os
import sys
import glob
import itertools as it
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
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

tags = enum('START', 'READY', 'DONE', 'FAILED', 'EXIT')


class experiment_handling(object):
    """Class doc string."""

    def __init__(self, sample_size, parameter_combinations, index, path_raw,
                 path_res='./data/'):
        """
        Set up the experiment handling class.

        saves the sample size and parameter combinations of the
        experiment and creates a list of tasks containing parameter
        combinations, ensemble index and path to the corresponding
        output files.

        also sets up the MPI environment to keep track of
        master and subordinate nodes.

        Parameters:
        -----------
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
        self.sample_size = sample_size
        self.parameter_combinations = parameter_combinations
        self.index = index

        # add "/" to paths if missing
        self.path_raw = path_raw + "/" if not path_raw.endswith("/") else\
            path_raw
        self.path_res = path_res + "/" if not path_res.endswith("/") else\
            path_res

        self.comm = MPI.COMM_WORLD
        self.status = MPI.Status()
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.master = 0
        self.nodes = range(1, self.size)
        self.n_nodes = self.size-1

        if self.rank == 0:
            self.amMaster = True
            self.amNode = False
        else:
            self.amMaster = False
            self.amNode = True

        # create list of tasks (parameter_combs*sample_size)
        # and paths to save the results.
        self.tasks = []
        for s in range(self.sample_size):
            for p in self.parameter_combinations:
                filename = self.path_raw + self._get_ID(p, s)
                if not os.path.exists(filename):
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
            The first p paramters need to fit to the parameter_combinations.
            The last parameter of run_func has to be named filename
            If run_func succeded, it returns >=0, else it returns < 0
        skipbadruns : bool (Default: False)
            If you don't want this function to check for bad runs that shall be
            recalculated, than set to "True". Possible reason: speed.
        """
        if self.amMaster:
            # check, if path exists. If not, create.
            if not os.path.exists(self.path_raw):
                os.makedirs(self.path_raw)

            # give brief feedback about remaining work.
            print str(len(self.tasks)) + " of "\
                + str(len(self.parameter_combinations)*self.sample_size)\
                + " single computations left"

            # check if nodes are available. If not, do serial calculation.
            if self.n_nodes < 1:
                print "Only one node available. No parallel execution."
                for task in self.tasks:
                    (params, filename) = task
                    result = -1
                    while result < 0:
                        result = run_func(*params, filename=filename)

            print "Splitting calculations to {} nodes.".format(self.n_nodes)
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
                    # node failed to complete. retry
                    old_task = n_return
                    self.comm.send(old_task, dest=source, tag=tags.START)
                elif tag == tags.DONE:
                    # node succeeded
                    result = n_return
                    tasks_completed += 1
                    self._progress_report(tasks_completed, len(self.tasks),
                                          "Calculating {} ..."
                                          .format(source))
                elif tag == tags.EXIT:
                    # node completed all tasks. close
                    closed_nodes += 1
            print "Calculating 0 ...done."

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
                    result = run_func(*params, filename=filename)
                    if result >= 0:
                        self.comm.send(result, dest=self.master, tag=tags.DONE)
                    else:
                        self.comm.send(task, dest=self.master, tag=tags.FAILED)
                elif tag == tags.EXIT:
                    break

            self.comm.send(None, dest=self.master, tag=tags.EXIT)

        self.comm.Barrier()

    def resave(self, eva, name):
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
        """
        # Prepare list of effective parameter combinations for MultiIndex
        eff_params = {self.index[k]: np.unique([p[k] for p in
                                                self.parameter_combinations])
                      for k in self.index.keys()}

        # if eva returns a data frame,
        # add the indices and column names to the list of effective parameters.
        eva_return = [eva[evakey](np.sort(glob.glob(self.path_raw +
                      self._get_ID(self.parameter_combinations[0]))))
                      for evakey in eva.keys()]

        if isinstance(eva_return[0], pd.core.frame.DataFrame) and\
                not isinstance(eva_return[0].index, pd.core.index.MultiIndex):
            eff_params['timesteps'] = eva_return[0].index.values
            eff_params['observables'] = eva_return[0].columns.values
            index_order = list(it.chain.from_iterable([self.index.values(),
                               ["timesteps", "observables"]]))
            input_is_dataframe = True
        else:
            input_is_dataframe = False

        if self.amMaster:
            print 'processing ', name
            print 'under operators ', eva.keys()
            # create save_path if it is not yet existing
            if not os.path.exists(self.path_res):
                os.makedirs(self.path_res)

            # Create MultiIndex and Dataframe
            mIndex = pd.MultiIndex.from_product(eff_params.values(),
                                                names=eff_params.keys())

            if input_is_dataframe:
                mIndex = mIndex.reorder_levels(index_order)

            df = pd.DataFrame(index=mIndex)

            task_index = 0
            tasks_completed = 0
            n_tasks = len(self.parameter_combinations)*len(eva.keys())
            closed_nodes = 0

            # Check if nodes are available. If not, do serial computation.
            if self.n_nodes < 1:
                print "Only one node available. No parallel execution."
                for task_index in range(n_tasks):
                    p_index, k_index = divmod(task_index, len(eva.keys()))
                    p, key = (self.parameter_combinations[p_index],
                              eva.keys()[k_index])
                    mx = tuple(p[k] for k in self.index.keys())
                    fnames = np.sort(glob.glob(self.path_raw + self._get_ID(p)))
                    print self._get_ID(p)
                    eva_return = eva[key](fnames)
                    if input_is_dataframe:
                        stack = pd.DataFrame(eva_return.stack(dropna=False))
                        stackIndex = pd.MultiIndex(levels=stack.index.levels,
                                                   labels=stack.index.labels,
                                                   names=[u'timesteps',
                                                          u'observables'])
                        eva_return = pd.DataFrame(stack,
                                                  index=stackIndex)\
                            .sortlevel(level=1)
                    df.loc[mx, evakey] = eva_return

            # If nodes are available, distribute work amongst nodes.

            while closed_nodes < self.n_nodes:
                # master keeps subordinate nodes buzzy:
                data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                      status=self.status)
                source = self.status.Get_source()
                tag = self.status.Get_tag()
                if tag == tags.READY:
                    # node ready to work.
                    if task_index < n_tasks:
                        # if there is work, distribute it
                        p_index, k_index = divmod(task_index, len(eva.keys()))
                        task = (self.parameter_combinations[p_index],
                                eva.keys()[k_index])
                        self.comm.send(task, dest=source, tag=tags.START)
                        task_index += 1
                    else:
                        # if not, release worker
                        self.comm.send(None, dest=source, tag=tags.EXIT)
                elif tag == tags.DONE:
                    (mx, key, eva_return) = data
                    df.loc[mx, evakey] = eva_return
                    tasks_completed += 1
                    self._progress_report(tasks_completed, n_tasks,
                                          "Post-processing {} ..."
                                          .format(source))
                elif tag == tags.EXIT:
                    closed_nodes += 1
            print 'Post-processing done'

        if self.amNode:
            # Nodes work as follows:
            name = MPI.Get_processor_name()
            while True:
                self.comm.send(None, dest=self.master, tag=tags.READY)
                task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG,
                                      status=self.status)
                tag = self.status.Get_tag()
                if tag == tags.START:
                    # go work:
                    (p, key) = task
                    mx = tuple(p[k] for k in self.index.keys())
                    fnames = np.sort(glob.glob(self.path_raw+self._get_ID(p)))
                    eva_return = eva[key](fnames)
                    if input_is_dataframe:
                        stack = pd.DataFrame(eva_return.stack(dropna=False))
                        stackIndex = pd.MultiIndex(levels=stack.index.levels,
                                                   labels=stack.index.labels,
                                                   names=[u'timesteps',
                                                          u'observables'])
                        eva_return = pd.DataFrame(stack,
                                                  index=stackIndex)\
                            .sortlevel(level=1)
                    self.comm.send((mx, key, eva_return), dest=self.master,
                                   tag=tags.DONE)
                elif tag == tags.EXIT:
                    break

            self.comm.send(None, dest=self.master, tag=tags.EXIT)

        self.comm.Barrier()

    @staticmethod
    def _get_ID(parameter_combination, i=None):
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
        res = res.replace(".", "o")  # replace dots with an "o"
        res = res.replace("'", "")  # remove 's from values of string variables
        # Remove all the other left over mean charakters that might fuck with you
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
        """
        A small progress report for a loop of defined length.

        Parameters
        ----------
        i : int
            the current position in the loop
        loop_length : int
            the length of the loop
        msg : string
            (optional) a preceding string
        """
        percent = str(int(np.around(float(i)/loop_length, 2) * 100))
        sys.stdout.write("\r")
        sys.stdout.flush()
        sys.stdout.write(msg + " [" + percent + "%] ")
        sys.stdout.flush()

        if i == loop_length-1:
            sys.stdout.write("\n")
            sys.stdout.flush()


def even_time_series_spacing(dfi, N, t0=None, tN=None):
    """
    Interpolate irregularly spaced time series.

    To obtain regularly spaced data.

    Parameters
    ----------
    dfi : pandas dataframe
        dataframe with one dimensional index containing
        time stamps as index values.
    N   : int
        number of regularly spaced timestamps in the
        resulting time series.
    t0  : float
        starting time of the resulting time series -
        defaults to the first time step of the input
        time series.
    tN  : float
        end time of the resulting time series -
        defaults to the last time step of the input
        time series.

    Returns
    -------
    dfo : pandas dataframe
        pandas dataframe with n regularly spaced
        time steps starting at t0 and ending at tN
        inclusively. Output data is interpolated from
        the input data.
    """
    if t0 is None:
        t0 = dfi.index.values[0]
    if tN is None:
        tN = dfi.index.values[-1]

    timestamps = np.linspace(t0, tN, N)

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

    data_points = {}
    data_points['time'] = timestamps
    for o in observables:
        x = [t if t < t_max else float('NaN') for t in timestamps]
        data_points[o] = interpolations[o](x)

    dfo = pd.DataFrame(data_points, index=timestamps)

    return dfo
