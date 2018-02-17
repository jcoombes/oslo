"""
A simple implementation of the Oslo model, built using test-driven development.

TODO:

task1: build a test/check test is sufficiently testy.
task2c: Scaling function, behaviour, how does h_tilde change with t during transient.
main_2d gives strange coeffs for quadratic,
take mean of crossover_times for a given system size. create a dict of system size, cutoff lookup values.
measure height averaged over time.
2d: does this match with average heights from main_2a/system size?

task2e: hooray, works. Can we do linear regression on our supermain2(main_2e, scaling=1) plot?
task2f: ask blaine how to reverse engineer r^2 values to fit linear regression. do <z>_t and sigma_z(t)
task2g: theoretical > data collapse > Experiment matches theory?
"""
import logbinsixfeb as lb
import numpy as np
import numpy.polynomial as poly  # Used to fit a polynomial to crossover time graph.
import scipy.optimize as opt  # Also used to fit polynomial to crossover time graph.
import collections  # We need a double ended queue for our tree-search implementation of relaxation() function.
import matplotlib.pyplot as plt
import seaborn as sns  # makes pretty figures
import datetime  # we want to uniquely name figures. I could use GUID, but I prefer this method.
import pickle
import scipy.stats #linear regression with pvalue, rvalue.
sns.set()

def drive(slopes):
    """

    :param slopes: a non-empty list of non-negative ints, slopes
    :return: slopes with a grain added at position 1.
    """
    if len(slopes) == 0: raise ValueError
    slopes[0] += 1
    return slopes


def relax(site, slopes):
    """
    Relax one particular site.

    :param site: int, index of which slope site to relax, zero index.
    :param slopes: list of non-negative ints.
    :return: slopes: list of non-negative ints after one site relaxation.
    """
    if len(slopes) == 0: raise ValueError
    if len(slopes) - 1 < site or site < 0: raise IndexError  # can take len(slopes) possible values

    if site == 0:
        slopes[0] -= 2
        slopes[1] += 1

    elif site < len(slopes) - 1:
        slopes[site - 1] += 1
        slopes[site] -= 2
        slopes[site + 1] += 1

    else:
        slopes[-2] += 1
        slopes[-1] -= 1

    return slopes


def thresh_update(site, thresh, p):
    """
    Updates the threshhold slope for a given site. Notice that we always call this immediately after thresh_update.

    :param site: int, where do we adjust threshhold slope?
    :param thresh: list of ints, threshhold slopes
    :param p: probability slope is 1.
    :return: thresh, updated at relevant site.
    """
    if len(thresh) == 0: raise ValueError
    if len(thresh) - 1 < site or site < 0: raise IndexError  # can take len(thresh) possible values

    if p > 1.0 or p < 0.0: raise ValueError

    if np.random.rand() <= p:
        thresh[site] = 1
    else:
        thresh[site] = 2
    return thresh


def relax_and_thresh(site, slopes, thresh, p):
    """
    This combines the relax() function and the thresh_update() function.
    Designed to prevent user errors from calling one function without the other.

    :param site:
    :param slopes:
    :param thresh:
    :param p:
    :return:
    """
    slopes = relax(site, slopes)
    thresh = thresh_update(site, thresh, p)
    return slopes, thresh


def relaxation(slopes, thresh, p):
    """
    Relaxes everything that can relax.
    Even though thresh isn't explicitly returned, it it updated as a *side-effect of mutable lists*.
    I think instead of using a loop in a loop, you can save on unnecessary comparisons by representing
    the as a binary tree. Then you can use tree traversal algorithms.

    :param slopes: list of ints.
    :param thresh: list of ints, either 1 or 2.
    :return: slopes, but after all the slopes that can relax, have relaxed.
    """
    if len(slopes) != len(thresh): raise IndexError

    hitlist = collections.deque([0])  # list of site indices on slope we would like to relax.
    s = 0

    while len(hitlist) > 0:
        index = hitlist.popleft()

        # print(index, sep="\t ")
        if slopes[index] > thresh[index]:
            s += 1
            # print(slopes)
            # print("relaxing index {} because {} > {}".format(index, slopes[index], thresh[index]))
            slopes, thresh = relax_and_thresh(index, slopes, thresh, p)

            hitlist.appendleft(index)

            if index > 0:
                hitlist.appendleft(index - 1)
            if index < len(slopes) - 1:  # len(slopes) is one larger than the maximum index because zero-indexing.
                # print(index, "<", len(slopes))
                hitlist.append(index + 1)
    return slopes, s


def relaxation_bifurcation(slopes, thresh, p):
    """
    Variant of relaxation function, used to build avalanche bifurcation diagrams.

    :param slopes: list of ints.
    :param thresh: list of ints, either 1 or 2.
    :return: slopes, but after all the slopes that can relax, have relaxed.
    """
    if len(slopes) != len(thresh): raise IndexError

    hitlist = collections.deque([0])  # list of indices on slope we would like to add.
    bifurcate = []

    while len(hitlist) > 0:
        index = hitlist.popleft()

        # print(index, sep="\t ")
        if slopes[index] > thresh[index]:
            bifurcate.append(index)
            print(len(bifurcate))
            # print(slopes)
            # print("relaxing index {} because {} > {}".format(index, slopes[index], thresh[index]))
            slopes, thresh = relax_and_thresh(index, slopes, thresh, p)

            hitlist.appendleft(index)

            if index > 0:
                hitlist.appendleft(index - 1)
            if index < len(slopes) - 1:  # len(slopes) is one larger than the maximum index because zero-indexing.
                # print(index, "<", len(slopes))
                hitlist.append(index + 1)
    return slopes, bifurcate


def relaxation_crossover(slopes, thresh, p):
    """
    Relaxes everything that can relax.
    I think instead of using a loop in a loop, you can save on unnecessary comparisons by representing
    the as a binary tree. Then you can use tree traversal algorithms.

    :param slopes: list of ints.
    :param thresh: list of ints, either 1 or 2. #updated as a side effect.
    :return: slopes, but after all the slopes that can relax, have relaxed.
    :return: cross, bool whether this relaxation caused the rightmost site to relax (did a grain to leave the system?)
    """
    if len(slopes) != len(thresh): raise IndexError

    hitlist = collections.deque([0])  # list of site indices on slope we would like to relax.
    s = 0
    cross = False

    while len(hitlist) > 0:
        index = hitlist.popleft()

        # print(index, sep="\t ")
        if slopes[index] > thresh[index]:
            s += 1
            # print(slopes)
            # print("relaxing index {} because {} > {}".format(index, slopes[index], thresh[index]))
            slopes, thresh = relax_and_thresh(index, slopes, thresh, p)

            hitlist.appendleft(index)

            if index > 0:
                hitlist.appendleft(index - 1)
            if index < len(slopes) - 1:  # len(slopes) is one larger than the maximum index because zero-indexing.
                # print(index, "<", len(slopes))
                hitlist.append(index + 1)
            elif index == len(slopes) - 1:
                cross = True
    return slopes, s, cross


def relax_and_thresh_init(size=4, p=0.5, seed=0):
    """
    Sets up all the parameters (for multiple "main" functions.)
    :return:
    """
    slopes = [0] * size
    np.random.seed(seed)
    thresh = [1 if rand <= p else 2 for rand in np.random.rand(size)]

    return slopes, thresh


def height(slopes):
    """
    Measures the height of the pile.
    :return: height, int.
    """
    return sum(slopes)


def main_bifurcation(size, p, t_max, seed):
    """
        Solves task 2a.
        'Starting from an empty system, measure and plot the total height of the pile
        as a function of time t for a range of system sizes.'

        :param size: int, how big is system?
        :param p: when p=1, all thresh is 1, when p =0, all thresh is 2.
        :param t_max: cast into int, how many grains do we add?
        :param seed: int, change this to generate different runs.
        :return: plot of heights with grains added.
        """
    slopes, thresh = relax_and_thresh_init(size, p, seed)

    bif = 'replaceme'
    for t in range(int(t_max)):
        slopes = drive(slopes)
        slopes, bif = relaxation_bifurcation(slopes, thresh, p)
        print(bif)

    return bif


def fiddling(size, p, t_max):
    """
    Trying to get a bifurcation image.
    :return:
    """
    biffy = main_bifurcation(size, p, t_max, seed=0)
    canvas = np.zeros((max(biffy) + 1, len(biffy)))

    print(canvas)

    for i in range(len(biffy)):
        canvas[biffy[i]][i] = 1

    sum(canvas)
    plt.imshow(canvas)
    return canvas


def moving_average(heights, window=25):
    """

    :param heights: list of ints, how high is ricepile.
    :param window: ints, how far in each direction to take values .
    :return: smooth_heights: the same thing, but  with the temporal average taken between [t-W] and t+W
    """

    def add_window(heights, site, window):
        """
        helper function. The main benefit is that we can feed it variable window sizes.
        slicing a list creates a new list, slowing performance.
        Would I rather use a for loop and iterate over this?

        :param heights: list of unsmoothed data
        :param site: integer, indexing heights, where is the centre of moving average.
        :param window:
        :return: windowed_sum
        """
        if window >= len(heights) / 2:
            window = int(len(heights) / 10)

        if site < window:
            pre_slice = heights[:2 * site + 1]
            # print("for site {}, pre-window is from {} to {}, len {}".format(site, 0, 2*site+1, len(pre_slice)))
            windowed_sum = sum(pre_slice) / len(pre_slice)
        elif site + window > len(heights) - 1:
            post_slice = heights[2 * site + 1 - len(heights):]
            # print("for site {}, post-window is from {} to {}, len {}".format(site, 2*site+1-len(heights), len(heights), len(post_slice)))
            windowed_sum = sum(post_slice) / len(post_slice)
        else:
            slice = heights[site - window: site + window + 1]
            # print("for site {}, window is from {} to {}, len {}".format(site, site-window, site+window+1, len(slice)))
            windowed_sum = sum(slice) / len(slice)
        return windowed_sum

    if not heights.any():
        return heights  # Deals with empty list edge case.

    smooth = []
    for site in range(len(heights)):
        smooth.append(add_window(heights, site, window))
    return smooth


def cross_estimate(size):
    """
    This should give an overestimate of crossover time, which leaves a little margin to start collecting data.
    :return: approximate value of the crossover time.
    """
    return 0.85 * size * (size + 1)

def time_estimate(t_max, size):
    """
    Returns a fitted estimate of how long it takes to drive and relax t_max times for a system of size 128.

    :param t_max:
    :return: how_many_seconds_does_it_take?
    """
    T_l128 = lambda t_max: 4.04776786e-04 * t_max - -3.785
    T_t10000 = lambda size: 0.02378528 * size + 0.095125

    return T_t10000(128) + 4.04776786e-04 * (t_max-10000) + 0.02378528 * (size-128)

def time_average(arr, start_time):

    """
    returns the time average of an arbitrary array.
    This takes an arbitrary array because I want to use this to make mean, and standard deviation function,
    Maybe I could use this for higher order moments as well.

    e.g. when arr=h,
    <h(t, L)>_t = lim T -> inf, 1/T * sum_t0^t0+T(h(t, L))

    :param arr: numpy array, what do you want to approximate the time average of?
    :param start_time: int or float,
    :return mean: int, the function averaged with time.
    """
    steady_state_arr = arr[start_time:]  # T implicitly defined here, as t_max(the length of arr) minus t_init.
    return steady_state_arr.sum(dtype=np.int64) / steady_state_arr.size


def standard_deviation(arr, start_time):
    """

    :param arr: what would you like to find the standard_deviation of?
    :param start_time: int, which timestep do you want to begin averaging from?
    :return standard_dev: float, the standard deviation.
    """
    arr_squared = arr * arr
    var = time_average(arr_squared, start_time) - time_average(arr, start_time) ** 2
    standard_dev = np.sqrt(var)
    return standard_dev


def picklification(size, p, t_max, seed, heights, slopes, thresh, filename='heights'):
    """

    :param seed: int
    :param size: int
    :param t_max: int
    :param heights: array[floats]
    :return: pickle string.
    """
    with open('pickle/' + str(filename) + '.pickle', 'wb') as f:
        picklestring = pickle.dumps(
            {"seed": seed, "p": p, "size": size, "t_max": t_max, "heights": heights, "slopes": slopes,
             "thresh": thresh})
        f.write(picklestring)
    return 'files written to pickle/{}.pickle'.format(filename)

def depicklification(filename):
    with open('pickle/' + str(filename) + '.pickle', 'rb') as f:
        run_bytes = f.read()
        run_dict = pickle.loads(run_bytes)
    return run_dict

def heights_measure(size, p, t_max, seed):

    """
    Measures the total height of the pile starting from an empty system.
    :return: heights at every timestep
    """
    slopes, thresh = relax_and_thresh_init(size, p, seed)
    heights = []

    for t in np.arange(t_max, dtype=np.int64):
        slopes = drive(slopes)
        slopes = relaxation(slopes, thresh, p)[0]
        heights.append(height(slopes))

    return heights, slopes, thresh

def pickle_cross(size, p, trials, seed, cross, filename='cross'):
    """
    Writes a pickle file with data.
    To read back this data use depicklification(filename)

    :param size: int,
    :param p: float, 0 < p < 1
    :param trials:
    :param seed:
    :param filename: Suggest using a combination of input parameters?
    :return: Success string, containing location of saved files.
    """
    with open('pickle/cross/' + str(filename) + '.pickle', 'wb') as f:
        picklestring = pickle.dumps(
            {"size" : size, "p" : p, "trials" : trials, "seed" : seed, "cross": cross}
            )
        f.write(picklestring)
    return 'files written to pickle/cross/{}.pickle'.format(filename)


def pickle_ava(size, p, tmax, seed, avalanches, slopes, thresh, cross, filename='ava'):
    """

    :param size:
    :param p:
    :param tmax:
    :param seed:
    :param avalanches:
    :param slopes:
    :param thresh:
    :param cross:
    :param filename:
    :return:
    """

    with open('pickle/ava/' + str(filename) + '.pickle', 'wb') as f:
        picklestring = pickle.dumps(
            {"size" : size, "p" : p, "tmax" : tmax, "seed" : seed, "cross": cross, \
             "avalanches" : avalanches, 'slopes' : slopes, 'thresh':thresh}
            )
        f.write(picklestring)
    return 'files written to pickle/ava/{}.pickle'.format(filename)


def cross_measure(sizes, p, trials, seed):
    """
    Initially the sizes list is going to have to be len(sizes) == 1, because the plot function can only take two vectors.
    :param sizes: list of ints, which system sizes would you like to plot?
    :param p: input parameter to oslo model.
    :param trials: how many t_c data points do you want for each system size.
    :return: return_size, list of ints, equivalent to [[size] * trials for size in sizes]
    :return: crossovers, list of crossover times.
    """
    return_sizes = []
    crossovers = []

    for size in sizes:
        for j in range(trials):
            print(j,end=" ")
            slopes, thresh = relax_and_thresh_init(size, p, seed + j)
            crossed = False
            tc = 0

            while not crossed:
                slopes = drive(slopes)
                slopes, crossed = relaxation_crossover(slopes, thresh, p)[0::2]
                tc += 1

            return_sizes.append(size)
            crossovers.append(tc)

    return return_sizes, crossovers

def avalanche_measure(size, p, t_max, seed):
    """
    Measures the avalanche size at every timestep.

    :param size:
    :param p:
    :param t_max:
    :param seed:
    :return: avalanches, slopes, thresh, crossover_time
    """

    slopes, thresh = relax_and_thresh_init(size, p, seed)
    avalanches = []
    already_crossed = False

    for t in np.arange(t_max, dtype=np.int64):
        slopes = drive(slopes)
        slopes, s, cross = relaxation_crossover(slopes, thresh, p)
        if not already_crossed and cross:
            crossover_time = t
            already_crossed = True
        avalanches.append(s)

    return avalanches, slopes, thresh, crossover_time

def main_2a_measure(size, p, t_max, seed):
    """
    Measures the total height of the pile starting from an empty system.
    :return: heights at every timestep
    """
    slopes, thresh = relax_and_thresh_init(size, p, seed)
    heights = []

    for t in range(int(t_max)):
        slopes = drive(slopes)
        slopes = relaxation(slopes, thresh, p)[0]
        heights.append(height(slopes))

    return heights


def main_2a_plot(heights, size, log, save, figname):
    """
    Plots the total height of the pile on either linear axes or loglog axes.

    :return: a lovely plot, also prints the average height once the system reaches the recurrent configurations.
    """
    crosstime = int(cross_estimate(size))
    avg = 0
    try:
        recurrent_heights = heights[crosstime:]  # This is a rough way to cut transient and recurrent configurations.
        avg = sum(recurrent_heights) / len(recurrent_heights)
        print("the average recurrent height of the system is " + str(avg))
    except IndexError:
        pass

    fig = plt.plot(heights, label="System Size: {}".format(size))
    plt.xlabel("Grains in system")
    plt.ylabel("Slope height after relaxation")
    plt.title("System height(time)")
    plt.legend()

    if log:
        plt.xscale("log")
        plt.yscale("log")
        if save:
            plt.savefig(figname + " loglog")
    else:
        plt.xscale("linear")
        plt.yscale("linear")
        if save:
            plt.savefig(figname)
    return fig, avg


def main_2a(run_dict, log=0, save=0, figname="Height(grains)"):
    """
    Solves task 2a.
    'Starting from an empty system, measure and plot the total height of the pile
    as a function of time t for a range of system sizes.'

    :param run_dict: run_dict is a dictionary with information about ...

     size: int, how big is system?
     p: when p=1, all thresh is 1, when p =0, all thresh is 2.
     t_max: cast into int, how many grains do we add?
     seed: int, change this to generate different runs.
    :param log: bool, generate linear plot or loglog plot?
    :param save: bool, do you want to save the figure?
    """
    size, p, t_max, seed, heights = run_dict['size'], run_dict['p'], run_dict['t_max'], run_dict['seed'], run_dict['heights']
    return main_2a_plot(heights, size, log, save, figname)


def main_2b():
    """
    Theoretical -
    show that for very large system sizes, <h>=<z>L where h is average pile height, z is mean slope, L is system size.
    show that the mean crossover time (time required for grain topple through the whole system) is <t_c>=<z>L(L+1)/2
    """
    pass


def main_2c_measure(run_dict):
    """
    This function will be very similar to 2a, except we have scaled the time and the height.
    :return: scaled times, array of floats.
    :return: scaled heights, array of floats. Hopefully these will be the same for multiple system sizes.
    """
    size, seed, heights = run_dict['size'], run_dict['seed'], run_dict['heights'] #Explicit is better than Implicit
    t_max, p, slopes, thresh = run_dict['t_max'], run_dict['p'], run_dict['slopes'], run_dict['thresh']

    scaled_times = np.arange(t_max)/size**2
    scaled_heights = np.array(heights)/size

    return scaled_times, scaled_heights


def main_2c_plot(save, sizelist, *args):
    """
    Plot our amazing data collapse, notice that because we have scaled the time axis we need to use a larger t_max
    in the main_2c_measure() function so that t_max/(size**2) is the same for all system sizes.
    :param: save, boolean, do you want to save this figure to current directory?
    :param: scaled_times, list of ints. zeroth positional argument.
    :param: scaled_heights, list of ints. first positional argument.
    :param: scaled_times2, list of ints. second positional argument.
    :param: scaled_heights2, list of ints. third positional argument.
    :param: scaled_times3, list of ints. fourth positional argument.
    :param: scaled_heights3, list of ints. fifth positional argument.
    ... etc, this function will just keep plotting all the times, and all the heights you give it.

    :return: a beautiful plot with an arbitrary number of systems in it.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel("scaled time t/L^2")
    ax.set_ylabel("scaled height h/L")
    ax.set_title("Data collapse")

    if len(args) % 2 != 0:
        raise (ValueError("Incorrect number of positional arguments"))

    else:
        for i in range(0, len(args), 2):
            ax.plot(args[i], args[i + 1], label = "System Size L: {}".format(sizelist[i]))
            ax.legend()

        if save:
            file_identifier = str(datetime.datetime.now()).replace(".", "-").replace(" ", "_").replace(":",
                                                                                                       "_")  # e.g. '2018-02-04__15:43:06-761532'
            fig.savefig("Data_Collapse_" + file_identifier + '.png', format='png')
        return fig


def main_2c(run_dicts):
    """
    Solves task 2c.

    Guided by your answers to the two questions in 2b, produce a data collapse for the processed height h^tilde(t,L)
    vs. time t for various system sizes. Explain carefully how you produced a data collapse
    and express that mathematically, introducing a scaling function F: h^tilde(t, L) = something F(argument),
    identifying 'something' and the 'argument'. How does the scaling function F(x) behave for large arguments x>>1
    and for small arguments x<<1 and why must it be so? From this result, obtain/predict how h^tilde increases
    as a function of t during the transient.

    I think because the mean height scales linearly with system size,
    and the average cross over time scales quadratically with system size,
    this is just a matter of linearly scaling height, and quadratically scaling time for different L.
    :param: sizes, list of system sizes to calculate and plot.
    :param: p, probability threshhold slope height is 1 rather than 2.
    :param: scaled_t_max, float t_max/(size^2). How many timesteps do you want the final data collapse to span.
    :param: seed, int where should the random generator start. Note that for non-deterministic runs you need to change this.
    :return: beautiful data collapse plot for an arbitrary number of systems.
    """
    sizelist = [] #Just used to help create the plot legend.
    time_height_pair_list = []  # Will contain scaled_time, scaled_height, scaled_time2, scaled_height2, etc...

    for run_dict in run_dicts:
        sizelist.append(run_dict['size'])
        sizelist.append(run_dict['size'])  # This duplication is because main_2c_plot loops over range(,,step=2)
        scaled_times, scaled_heights = main_2c_measure(run_dict)
        scaled_heights = moving_average(scaled_heights)
        time_height_pair_list.append(scaled_times)
        time_height_pair_list.append(scaled_heights)

    return_fig = main_2c_plot(1, sizelist,*time_height_pair_list)
    return return_fig


def main_2d_measure(sizes, p, trials, seed):
    """
    Initially the sizes list is going to have to be len(sizes) == 1, because the plot function can only take two vectors.
    :param syzes: list of ints, which system sizes would you like to plot?
    :param p: input parameter to oslo model.
    :param trials: how many t_c data points do you want for each system size.
    :return: syze, list of ints, what is the system size under test.
    :return: crossovers, list of crossover times.
    """
    syzes = []
    crossovers = []

    for size in sizes:
        for j in range(trials):
            slopes, thresh = relax_and_thresh_init(size, p, seed + j)
            crossed = False
            tc = 0

            while not crossed:
                slopes = drive(slopes)
                slopes, crossed = relaxation_crossover(slopes, thresh, p)[0::2]
                tc += 1

            syzes.append(size)
            crossovers.append(tc)

    return syzes, crossovers


def main_2d_plot(syzes, crossovers, fig=None, ax=None):
    """
    Plots the mean cross-over time for a range of system sizes.
    This seems to me like a ax.plot() with dots instead of lines.
    :param syzes: list of int, system sizes.
    :param crossovers: list of float, cross-over times.
    :return: lovely plot of arbitrary number of system sizes.
    """
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel("system size")
    ax.set_ylabel("crossover time")
    ax.set_title("Crossover time as a function of system size scatter plot")
    ax.scatter(syzes, crossovers)

    fig.savefig("crossover_time as a function of system size.png")
    return fig, ax


def main_2d(sizes=[4, 8, 16, 32, 64, 128, 256], p=0.5, trials=3, seed=0):
    """
    Solves task 2d.

    'Numerically measure the cross-over time, t_c(L) as the number of grains in the system before an added grain
    induces a grain to leave the system for the first time, starting from an empty system
    estimate the average cross-over time as <t_c(L)>. Demonstrate whether your data corroborate your theoretical prediction.

     :param: sizes, list of system sizes to calculate and plot.
     :param: p, probability threshhold slope height is 1 rather than 2.
     :param: trials, int how many data points would you like for each system size?
     :param: seed, int where should the random generator start. Note that for non-deterministic runs you need to change this.
     :return: cross-over time plot for an arbitrary number of systems. Plotted with <z>L**2(1+1/L)/2 theoretical reference.
     """

    if isinstance(sizes, (int, float)):
        sizes = [sizes]

    sizelist, crossovers = main_2d_measure(sizes, p, trials, seed)
    sizelist_array = np.array(sizelist)
    crossovers_array = np.array(crossovers)

    polynomial_coefficients, full = poly.polynomial.polyfit(sizelist_array, crossovers_array, 2, full=True)

    empirical_x = np.array(np.linspace(0, 260, 261))
    empirical_y = poly.polynomial.polyval(empirical_x, polynomial_coefficients)
    residuals = full[0]

    fig, ax = main_2d_plot(sizelist, crossovers)
    ax.plot(empirical_x, empirical_y, 'r-')
    legend2 = "polynomial with coefficients {} and least squares residual of {}".format(polynomial_coefficients,
                                                                                        residuals)

    return fig, polynomial_coefficients, residuals


def main_2d_ode(sizes=[4, 8, 16, 32, 64, 128], p=0.5, trials=3, seed=0):
    """
    Solves task 2d.

    'Numerically measure the cross-over time, t_c(L) as the number of grains in the system before an added grain
    induces a grain to leave the system for the first time, starting from an empty system
    estimate the average cross-over time as <t_c(L)>. Demonstrate whether your data corroborate your theoretical prediction.

     :param: sizes, list of system sizes to calculate and plot.
     :param: p, probability threshhold slope height is 1 rather than 2.
     :param: trials, int how many data points would you like for each system size
     :param: seed, int where should the random generator start. For non-deterministic runs you need to change this.
     :return: cross-over time plot for an arbitrary number of systems. Plotted with <z>L**2(1+1/L)/2 theoretical reference.
     """

    if isinstance(sizes, (int, float)):
        sizes = [sizes]

    sizelist, crossovers = main_2d_measure(sizes, p, trials, seed)
    sizelist_array = np.array(sizelist)  # e.g. [8, 8, 8, 16, 16, 16]. Think of this as x-coords for your graph.
    crossovers_array = np.array(crossovers)

    def quadratic_with_no_contant(x, a):
        return a * x ** 2 + a * x

    coefficients, covariance = opt.curve_fit(quadratic_with_no_contant, sizelist_array, crossovers_array, p0=(0.86))

    # polynomial_coefficients, full = poly.polynomial.polyfit(sizelist_array, crossovers_array, 2, full=True)

    empirical_x = np.array(np.linspace(0, 260, 261))
    empirical_y = quadratic_with_no_contant(empirical_x, *coefficients)

    fig = main_2d_plot(sizelist, crossovers)
    ax = fig.gca()
    ax.plot(empirical_x, empirical_y, 'r-')
    legend2 = "polynomial with coefficients {} and least squares residual of {}".format(coefficients, covariance)

    return fig, coefficients, covariance


def main_2e_measure(run_dict, start_time):
    """

    size, start_time, p, t_max, seed
    :return: time average plot.
    """

    heights = np.array(run_dict['heights'])
    t_av = time_average(heights, start_time)
    return t_av


def main_2e_plot(size, h_av_t, fig=None, ax=None):
    """
    main_2c_plot took a functional approach with an arbitrary number of x, y pairs as inputs.
    This function tries to use matplotlib's object oriented API rather than pyplot.
    If called with figure=None, ax =None it will create a figure and an axis to

    :param h_av_t: heights averaged in time. height[i] corresponds to the (start_time + i)th timestep.
    :param prob:
    :param size:
    :param figure: bool, would you like to print to a particular figure?
    :param ax: bool,
    :return:
    """
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1)

    ax.scatter(size, h_av_t, label='System Size: {}'.format(size))
    ax.set_xlabel("System Size")
    ax.set_ylabel("Time-average height of recurrent configurations")
    ax.set_title("Corrections to scaling.")

    return fig, ax


def main_2e(run_dicts, scaling=1):
    """
    Solves task 2e.

    'Now we consider the numerical data for the average height <h(t, L)> to investigate whether it contains signs of
    corrections to scaling. Assume the following form of the corrections to scaling
    <h(t, L)> = a_0L(1 - a_1 L^{-w1} + a_2 L^{-w2} + ...)
    where w1 >0, w2 > 0, and a_i are constants. Neglecting terms with i>1, estimate a0 and w1 using your measured data.'

    sizes=[4, 8, 16, 32, 64, 128], p=0.5, trials=3, seed=0

    :return: fig - some kind of graph.
    :return: estimates, tuple containing floats, (a0, w)
    """
    fig = None
    ax = None
    sizelist = []
    h_av_tlist = []

    for run_dict in run_dicts:
        size = run_dict["size"]
        start_time = int(cross_estimate(size))
        h_av_t = main_2e_measure(run_dict, start_time)
        if not scaling:
            fig, ax = main_2e_plot(size, h_av_t, fig, ax)

        sizelist.append(size)
        h_av_tlist.append(h_av_t)

    sizelist_array = np.array(sizelist) #L
    h_av_tlist_array = np.array(h_av_tlist) #<h(t, L)> = a0L (1-a1L^{-w1})

    if scaling:
        fig2, ax2 = plt.subplots(1, 1)

        a0 = np.linspace(1.728, 1.736, 9) #This breaks if start < 1.728 because log(x<0)
        r = []
        for a0_potential in a0:
            eks = sizelist_array
            why = a0_potential -  h_av_tlist_array/sizelist_array
            r.append(scipy.stats.linregress(np.log10(eks), np.log10(why)))


            ax2.loglog(eks, why,'o', label="a0: {:.4}".format(a0_potential))
        ax2.set_xlabel("system size")
        ax2.set_ylabel("a0-<h>/L")
        ax2.legend()

        return fig2, ax2, r #r[-3] is the closest fit, a0=1.734!

    else:

        def fit_func(l, a0):
            return a0 * l

        coefficients, covariance = opt.curve_fit(fit_func, sizelist_array, h_av_tlist_array)

        # polynomial_coefficients, full = poly.polynomial.polyfit(sizelist_array, crossovers_array, 2, full=True)

        empirical_x = np.array(np.linspace(0, max(sizelist)+5, max(sizelist)+6))
        empirical_y = fit_func(empirical_x, *coefficients)
        # residuals = full[0]

        ax.plot(empirical_x, empirical_y, label="Empirical fit slope: {:.3}".format(coefficients[0]))
        ax.legend()

        return fig, coefficients

def main_2f_measure(run_dict):
    return None


def main_2f_plot(ax, fig):
    return None


def main_2f(run_dicts, scaling = 1):
    """

    :param run_dicts:
    :param scaling: do you want to return the scaled linear regression (1) or original data (0)?
    :return: fig, ax, a plot of data.
    """

    fig = None
    ax = None
    sizelist = []
    h_av_tlist = []
    stdev_list = []

    for run_dict in run_dicts:
        size = run_dict["size"]
        start_time = int(cross_estimate(size))
        h_av_t = main_2e_measure(run_dict, start_time)
        stdev = standard_deviation(np.array(run_dict["heights"]), start_time)

        sizelist.append(size)
        h_av_tlist.append(h_av_t)
        stdev_list.append(stdev)
        sizearray = np.array(sizelist)
        stdev_array = np.array(stdev_list)

    if scaling == 1:
        fig, ax = plt.subplots(1,1)

        eks = np.log10(sizearray)
        r = scipy.stats.linregress(eks, stdev_array)
        fit = np.polynomial.polynomial.polyval(eks, [r.intercept, r.slope]) #log fit.
        #inverse_poly = lambda L,w, c: L**(1/w)+c
        #p_fit, more = scipy.optimize.curve_fit(inverse_poly, sizearray, stdev_array, p0=[2, -0.4])
        plo = 1

        if plo:
            #ax.plot(sizearray, stdev_list, label='data')
            #ax.plot(sizearray, inverse_poly(sizearray, *p_fit),label='L**1/{} + {}'.format(p_fit[0],p_fit[1]))
            ax.plot(eks, stdev_list, label='data')
            ax.plot(eks, fit, label = 'sigma = {:.3}L{:.3}'.format(r.slope, r.intercept))
            ax.plot(eks, np.polynomial.polynomial.polyval(eks, [0, 1]), label='simpler_hypothesis')

            #ax.set_xlabel('System Size')
            #ax.set_title('inverse polynomial fit.')

            ax.set_xlabel("log_10(System Size)")
            ax.set_ylabel("Standard Deviation of time average height")
            ax.set_title("Semilog fit to sigma_h(L) | r^2: {}".format(r.rvalue**2))

        else:
            #ax.plot(sizearray, stdev_array - inverse_poly(sizearray, *p_fit))
            #ax.set_xlabel('System Size')
            #ax.set_title('Inverse polynomial residual fit')

            ax.plot(eks, stdev_array - fit, label='residuals')
            ax.set_xlabel('log_10(system size)')
            ax.set_ylabel('data-fit')
            ax.set_title('residual fit')

        ax.legend()

        return fig, ax, r


    else:
        fig, ax = plt.subplots(1,1)

        ax.plot(sizelist, stdev_list, 'bo')
        ax.set_xlabel("System Size")
        ax.set_ylabel("Standard Deviation of time average height")
        ax.set_title("Task 2f")



    return fig, ax


def main_2g_measure(run_dict):
    """
    Measure the height distribution.
    This include the transient and the recurrent configurations.

    :param size: see main_measure_2a
    :param p: see main_measure_2a
    :param t_max: see main_measure_2a
    :param seed: see main_measure_2a
    :return sorted_unique_heights: array, sorted with duplicate elements removed.
    :return prob[heights]: normalised array of floats. Fraction of available configurations with height h

    expected usage example: plt.plot(main_2g_measure)
    """

    heights = run_dict['heights']
    sorted_unique_heights, counts = np.unique(heights, return_counts=True)
    prob = counts / counts.sum()
    return sorted_unique_heights, prob


def main_2g_plot(heights, prob, size, fig=None, ax=None):
    """
    main_2c_plot took a functional approach with an arbitrary number of x, y pairs as inputs.
    This function tries to use matplotlib's object oriented API rather than pyplot.
    If called with figure=None, ax =None it will create a figure and an axis to

    :param heights:
    :param prob:
    :param size:
    :param figure: bool, would you like to print to a particular figure?
    :param ax: bool,
    :return:
    """
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1)

    ax.plot(heights, prob, label='System Size: {}'.format(size))
    ax.set_xlabel("Height")
    ax.set_ylabel("Relative Probability")
    ax.set_title("Probability the system has a given height.")
    ax.legend()

    return fig, ax


def main_2g(run_dicts):
    """
    This is going to produce a lovely plot of relative likeliness of
    a configuration having a given height.

    :param size:
    :param p:
    :param t_max:
    :param seed:
    :return:
    """
    fig = None
    ax = None

    for run_dict in run_dicts:
        heights, count = main_2g_measure(run_dict)
        fig, ax = main_2g_plot(heights, count, run_dict['size'], fig, ax)

    return fig


def main_3a():
    raise NotImplementedError


def supermain(func, *args, **kwargs):
    """
    This is a handler for main_2a, main_2c, main_2e, main_2f, main_2g
    :param func:
    :return:
    """

    run_8, run_16, run_32, run_64 =       depicklification("8051e6_0"),       depicklification("16051e6_0"), \
                                          depicklification("32051e6_0"),      depicklification("64051e6_0")
    run_128, run_256, run_512, run_1024 = depicklification("128051000000_0"), depicklification("256051000000_0"), \
                                          depicklification("512051000000_0"), depicklification("1024051000000_0")

    run_dicts = [run_8, run_16, run_32, run_64, run_128, run_256, run_512, run_1024]
    try:
        return func(run_dicts, *args, **kwargs)
    except TypeError: #This is a slight bodge because main_2a only accepts one dict at a time.
        fig2a = None
        for run_dict in run_dicts:
            fig2a = func(run_dict, *args, **kwargs)
        return fig2a


def supermain2(func, *args, **kwargs):
    """
    same as supermain but with a sample size of 1e7 rather than 1e6.
    Gathers a whole bunch of run_dicts from local files and tries to run them.

    :param func: main_2a, main_2c, main_2e, main_2f, main_2g
    :param args:
    :param kwargs:
    :return: output of func.
    """

    run_8, run_16, run_32, run_64 =       depicklification("81e7_5"),       depicklification("161e7_5"), \
                                          depicklification("321e7_5"),      depicklification("641e7_5")
    run_128, run_256, run_512, run_1024 = depicklification("1281e7_5"), depicklification("2561e7_5"), \
                                          depicklification("5121e7_6"), depicklification("10241e7_5")

    run_dicts = [run_8, run_16, run_32, run_64, run_128, run_256, run_512, run_1024]
    try:
        return func(run_dicts, *args, **kwargs)
    except TypeError: #This is a slight bodge because main_2a only accepts one dict at a time.
        if not 'main_2a' in str(func):
            raise TypeError
        fig2a = None
        for run_dict in run_dicts:
            fig2a = func(run_dict, *args, **kwargs)
        return fig2a


if __name__ == "__main__":
    supermain(main_2a)
    supermain(main_2c)
    supermain(main_2d)
    supermain(main_2e)
    #supermain(main_2f)
    supermain(main_2g)

    #supermain3(main_3a)
    #supermain3(main_3b)
    #supermain(main_3c)