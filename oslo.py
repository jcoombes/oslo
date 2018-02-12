"""
A simple implementation of the Oslo model, built using test-driven development.

TODO:
task1: build a test/check test is sufficiently testy.
task2c: Scaling function, behaviour, how does h_tilde change with t during transient.
main_2d gives strange coeffs for quadratic,
take mean of crossover_times for a given system size. create a dict of system size, cutoff lookup values.
measure height averaged over time.
2d: does this match with average heights from main_2a/system size?

task2e: We have a gorgeous linear fit. This is a first approximation for a0
Can we use a more sophisticated fitting function to find w1, a1, a0?
Change the corrections to scaling graph to make the line labels more useful, how about the number of timesteps averaged over t_max-start_time?

task2f: all of it.
task2g: theoretical > data collapse > Experiment matches theory?
"""
import numpy as np
import numpy.polynomial as poly #Used to fit a polynomial to crossover time graph.
import scipy.optimize as opt #Also used to fit polynomial to crossover time graph.
import collections #We need a double ended queue for our tree-search implementation of relaxation() function.
import matplotlib
import matplotlib.pyplot as plt
import datetime #we want to uniquely name figures.

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
    if len(slopes)-1 < site or site < 0: raise IndexError # can take len(slopes) possible values

    if site == 0:
        slopes[0] -= 2
        slopes[1] += 1

    elif site < len(slopes)-1:
        slopes[site-1] += 1
        slopes[site] -= 2
        slopes[site+1] += 1

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
    if len(thresh)-1 < site or site < 0: raise IndexError # can take len(thresh) possible values

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
    slopes = relax(site,slopes)
    thresh = thresh_update(site, thresh, p)
    return slopes, thresh

def relaxation(slopes, thresh, p):
    """
    Relaxes everything that can relax.
    I think instead of using a loop in a loop, you can save on unnecessary comparisons by representing
    the as a binary tree. Then you can use tree traversal algorithms.

    :param slopes: list of ints.
    :param thresh: list of ints, either 1 or 2.
    :return: slopes, but after all the slopes that can relax, have relaxed.
    """
    if len(slopes) != len(thresh): raise IndexError

    hitlist = collections.deque([0]) #list of site indices on slope we would like to relax.
    s = 0

    while len(hitlist) > 0:
        index = hitlist.popleft()

        #print(index, sep="\t ")
        if slopes[index] > thresh[index]:
            s += 1
            #print(slopes)
            #print("relaxing index {} because {} > {}".format(index, slopes[index], thresh[index]))
            slopes, thresh = relax_and_thresh(index, slopes, thresh, p)

            hitlist.appendleft(index)

            if index > 0:
                hitlist.appendleft(index-1)
            if index < len(slopes)-1: #len(slopes) is one larger than the maximum index because zero-indexing.
                #print(index, "<", len(slopes))
                hitlist.append(index+1)
    return slopes, s

def relaxation_bifurcation(slopes, thresh, p):
    """
    Variant of relaxation function, used to build avalanche bifurcation diagrams.

    :param slopes: list of ints.
    :param thresh: list of ints, either 1 or 2.
    :return: slopes, but after all the slopes that can relax, have relaxed.
    """
    if len(slopes) != len(thresh): raise IndexError

    hitlist = collections.deque([0]) #list of indices on slope we would like to add.
    bifurcate = []

    while len(hitlist) > 0:
        index = hitlist.popleft()

        #print(index, sep="\t ")
        if slopes[index] > thresh[index]:
            bifurcate.append(index)
            print(len(bifurcate))
            #print(slopes)
            #print("relaxing index {} because {} > {}".format(index, slopes[index], thresh[index]))
            slopes, thresh = relax_and_thresh(index, slopes, thresh, p)

            hitlist.appendleft(index)

            if index > 0:
                hitlist.appendleft(index-1)
            if index < len(slopes)-1: #len(slopes) is one larger than the maximum index because zero-indexing.
                #print(index, "<", len(slopes))
                hitlist.append(index+1)
    return slopes, bifurcate

def relaxation_crossover(slopes, thresh, p):
    """
    Relaxes everything that can relax.
    I think instead of using a loop in a loop, you can save on unnecessary comparisons by representing
    the as a binary tree. Then you can use tree traversal algorithms.

    :param slopes: list of ints.
    :param thresh: list of ints, either 1 or 2.
    :return: slopes, but after all the slopes that can relax, have relaxed.
    :return: cross, bool whether this relaxation caused the rightmost site to relax (did a grain to leave the system?)
    """
    if len(slopes) != len(thresh): raise IndexError

    hitlist = collections.deque([0]) #list of site indices on slope we would like to relax.
    s = 0
    cross = False

    while len(hitlist) > 0:
        index = hitlist.popleft()

        #print(index, sep="\t ")
        if slopes[index] > thresh[index]:
            s += 1
            #print(slopes)
            #print("relaxing index {} because {} > {}".format(index, slopes[index], thresh[index]))
            slopes, thresh = relax_and_thresh(index, slopes, thresh, p)

            hitlist.appendleft(index)

            if index > 0:
                hitlist.appendleft(index-1)
            if index < len(slopes)-1: #len(slopes) is one larger than the maximum index because zero-indexing.
                #print(index, "<", len(slopes))
                hitlist.append(index+1)
            elif index == len(slopes)-1:
                cross = True
    return slopes, s, cross

def relax_and_thresh_init(size = 4, p = 0.5, seed = 0):
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
    biffy = main_bifurcation(size, p, t_max, seed = 0)
    canvas = np.zeros((max(biffy)+1, len(biffy)))

    print(canvas)

    for i in range(len(biffy)):
        canvas[biffy[i]][i] = 1

    sum(canvas)
    plt.imshow(canvas)
    return canvas

def moving_average(heights, window = 25):
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
        if window >= len(heights)/2:
            window = int(len(heights)/10)

        if site < window:
            pre_slice = heights[:2*site+1]
            #print("for site {}, pre-window is from {} to {}, len {}".format(site, 0, 2*site+1, len(pre_slice)))
            windowed_sum = sum(pre_slice)/len(pre_slice)
        elif site + window > len(heights) - 1:
            post_slice = heights[2*site+1-len(heights):]
            #print("for site {}, post-window is from {} to {}, len {}".format(site, 2*site+1-len(heights), len(heights), len(post_slice)))
            windowed_sum = sum(post_slice)/len(post_slice)
        else:
            slice = heights[site - window: site + window + 1]
            #print("for site {}, window is from {} to {}, len {}".format(site, site-window, site+window+1, len(slice)))
            windowed_sum = sum(slice)/len(slice)
        return windowed_sum

    if not heights:
        return heights #Deals with empty list edge case.

    smooth = []
    for site in range(len(heights)):
        smooth.append(add_window(heights, site, window))
    return smooth

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
    steady_state_arr = arr[start_time:] #T implicitly defined here, as t_max(the length of arr) minus t_init.
    return steady_state_arr.sum()/steady_state_arr.size

def standard_deviation(arr, start_time):
    """

    :param arr: what would you like to find the standard_deviation of?
    :param start_time: int, which timestep do you want to begin averaging from?
    :return standard_dev: float, the standard deviation.
    """
    arr_squared = arr*arr
    var = time_average(arr_squared, start_time) - time_average(arr)**2
    standard_dev = np.sqrt(var)
    return standard_dev

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

def main_2a_plot(heights, log, save, figname):
    """
    Plots the total height of the pile on either linear axes or loglog axes.

    :return: a lovely plot, also prints the average height once the system reaches the recurrent configurations.
    """
    recurrent_heights = heights[2000:]  # This is a rough way to cut transient and recurrent configurations.
    avg = sum(recurrent_heights)/ len(recurrent_heights)
    print("the average recurrent height of the system is " + str(avg))

    fig = plt.plot(heights)
    plt.xlabel("Grains in system")
    plt.ylabel("Slope height after relaxation")

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

def main_2a(size = 32, p = 0.5, t_max = 1e5, seed = 0, log=0, save = 0, figname="Height(grains)"):
    """
    Solves task 2a.
    'Starting from an empty system, measure and plot the total height of the pile
    as a function of time t for a range of system sizes.'

    :param size: int, how big is system?
    :param p: when p=1, all thresh is 1, when p =0, all thresh is 2.
    :param t_max: cast into int, how many grains do we add?
    :param seed: int, change this to generate different runs.
    :param log: bool, generate linear plot or loglog plot?
    :param save: bool, do you want to save the figure?
    """
    heights = main_2a_measure(size, p, t_max, seed)
    return main_2a_plot(heights, log, save, figname)

def main_2b():
    """
    Theoretical -
    show that for very large system sizes, <h>=<z>L where h is average pile height, z is mean slope, L is system size.
    show that the mean crossover time (time required for grain topple through the whole system) is <t_c>=<z>L(L+1)/2
    """
    pass

def main_2c_measure(size, p, t_max, seed):
    """
    This function will be very similar to 2a, except we have scaled the time and the height.
    :return: scaled times, list of floats.
    :return: scaled heights, list of floats. Hopefully these will be the same for multiple system sizes.
    """
    slopes, thresh = relax_and_thresh_init(size, p, seed)
    size_sq = size*size
    scaled_times = []
    scaled_heights = []

    for t in range(int(t_max)):
        scaled_times.append(t/size_sq)
        slopes = drive(slopes)
        slopes = relaxation(slopes, thresh, p)[0]
        scaled_heights.append(height(slopes)/size)


    return scaled_times, scaled_heights

def main_2c_plot(save, *args):
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

    if len(args)%2 != 0:
        raise(ValueError("Incorrect number of positional arguments"))

    else:
        for i in range(0, len(args), 2):
            ax.plot(args[i], args[i+1])
            ax.legend()

        if save:
            file_identifier = str(datetime.datetime.now()).replace(".", "-").replace(" ","_").replace(":","_") #e.g. '2018-02-04__15:43:06-761532'
            fig.savefig("Data_Collapse_" + file_identifier +'.png', format='png')
        return fig

def main_2c(sizes=[4, 8, 16, 32, 64, 128], p = 0.5, scaled_t_max = 1e5, seed=0):
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

    if isinstance(sizes, (int, float)):
        sizes = [sizes]

    time_height_pair_list = [] #Will contain scaled_time, scaled_height, scaled_time2, scaled_height2, etc...

    for size in sizes:
        t_max = int(size * size * scaled_t_max)
        print("Calculating for system size {} over {} timesteps".format(size, t_max))
        scaled_times, scaled_heights = main_2c_measure(size, p, t_max, seed)
        scaled_heights = moving_average(scaled_heights)
        time_height_pair_list.append(scaled_times)
        time_height_pair_list.append(scaled_heights)

    return_fig = main_2c_plot(1, *time_height_pair_list)
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
    syzes =  []
    crossovers = []

    for size in sizes:
        for j in range(trials):
            slopes, thresh = relax_and_thresh_init(size, p, seed+j)
            crossed = False
            tc = 0

            while not crossed:
                slopes = drive(slopes)
                slopes, crossed = relaxation_crossover(slopes, thresh, p)[0::2]
                tc += 1

            syzes.append(size)
            crossovers.append(tc)

    return syzes, crossovers

def main_2d_plot(syzes, crossovers):
    """
    Plots the mean cross-over time for a range of system sizes.
    This seems to me like a ax.plot() with dots instead of lines.
    :param syzes: list of int, system sizes.
    :param crossovers: list of float, cross-over times.
    :return: lovely plot of arbitrary number of system sizes.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel("system size")
    ax.set_ylabel("crossover time")
    ax.set_title("Crossover time as a function of system size scatter plot")
    ax.plot(syzes, crossovers, 'b+')

    fig.savefig("crossover_time as a function of system size.png")
    return fig

def main_2d(sizes=[4, 8, 16, 32, 64, 128, 256], p=0.5, trials = 3, seed=0):
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

    empirical_x = np.array(np.linspace(0, 260,  261))
    empirical_y = poly.polynomial.polyval(empirical_x, polynomial_coefficients)
    residuals = full[0]

    fig = main_2d_plot(sizelist,  crossovers)
    ax = fig.gca()
    ax.plot(empirical_x, empirical_y,'r-')
    legend2 = "polynomial with coefficients {} and least squares residual of {}".format(polynomial_coefficients, residuals)

    return fig, polynomial_coefficients,residuals

def main_2d_ode(sizes=[4, 8, 16, 32, 64, 128], p=0.5, trials = 3, seed=0):
    """
    Solves task 2d.

    'Numerically measure the cross-over time, t_c(L) as the number of grains in the system before an added grain
    induces a grain to leave the system for the first time, starting from an empty system
    estimate the average cross-over time as <t_c(L)>. Demonstrate whether your data corroborate your theoretical prediction.

     :param: sizes, list of system sizes to calculate and plot.
     :param: p, probability threshhold slope height is 1 rather than 2.
     :param: trials, int how many data points would you like for each system size
     :param: seed, int where should the random generator start. Note that for non-deterministic runs you need to change this.
     :return: cross-over time plot for an arbitrary number of systems. Plotted with <z>L**2(1+1/L)/2 theoretical reference.
     """

    if isinstance(sizes, (int, float)):
        sizes = [sizes]

    sizelist, crossovers = main_2d_measure(sizes, p, trials, seed)
    sizelist_array = np.array(sizelist) #e.g. [8, 8, 8, 16, 16, 16]. Think of this as x-coords for your graph.
    crossovers_array = np.array(crossovers)

    def quadratic_with_no_contant(x, a):
        return a*x**2 + a*x

    coefficients, covariance = opt.curve_fit(quadratic_with_no_contant, sizelist_array, crossovers_array, p0=(0.86))

    #polynomial_coefficients, full = poly.polynomial.polyfit(sizelist_array, crossovers_array, 2, full=True)

    empirical_x = np.array(np.linspace(0, 260,  261))
    empirical_y = quadratic_with_no_contant(empirical_x, *coefficients)
    #residuals = full[0]

    fig = main_2d_plot(sizelist,  crossovers)
    ax = fig.gca()
    ax.plot(empirical_x, empirical_y,'r-')
    legend2 = "polynomial with coefficients {} and least squares residual of {}".format(coefficients, covariance)

    return fig, coefficients, covariance


def main_2e_measure(size, start_time, p, t_max, seed):
    """

    :return: time average plot.
    """

    heights = np.array(main_2a_measure(size, p, int(t_max), seed)) #heights[i] is the height at timestep i
    t_av = time_average(heights, start_time)
    return t_av

def main_2e_plot():
    raise NotImplementedError

def main_2e(sizes=[4, 8, 16, 32, 64, 128], p=0.5, trials = 3, seed=0):
    """
    Solves task 2e.

    'Now we consider the numerical data for the average height <h(t, L)> to investigate whether it contains signs of
    corrections to scaling. Assume the following form of the corrections to scaling
    <h(t, L)> = a_0L(1 - a_1 L^{-w1} + a_2 L^{-w2} + ...)
    where w1 >0, w2 > 0, and a_i are constants. Neglecting terms with i>1, estimate a0 and w1 using your measured data.'

    :return: fig - some kind of graph.
    :return: estimates, tuple containing floats, (a0, w)
    """
    raise NotImplementedError
    h_av_t = main_2e_measure(t, L, T)
    fig = main_2e_plot()

    estimates = (0,0)

    return fig, estimates

def main_2g_measure(size, p, t_max, seed):
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

    heights = main_2a_measure(size, p, t_max, seed)
    sorted_unique_heights, counts = np.unique(heights, return_counts=True)
    prob = counts/counts.sum()
    return sorted_unique_heights, prob

def main_2g_plot(heights, prob, size, fig = None, ax=None):
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


def main_2g(sizes, p, t_max, seed):
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

    for size in sizes:
        heights, count = main_2g_measure(size, p, t_max, seed)
        fig, ax = main_2g_plot(heights, count, size, fig, ax)

    return fig



if __name__ == "__main__":
    main(4, 0.5)