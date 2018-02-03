"""
A simple implementation of the Oslo model, built using test-driven development.

TODO:
def main_2c

"""
import numpy as np
import collections
import matplotlib.pyplot as plt

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

    hitlist = collections.deque([0]) #list of indices on slope we would like to add.
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
    for i in range(int(t_max)):
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
    :return: heights, prints out average recurrent height also produces a plot.
    """
    slopes, thresh = relax_and_thresh_init(size, p, seed)
    heights = []

    for i in range(int(t_max)):
        slopes = drive(slopes)
        slopes = relaxation(slopes, thresh, p)[0]
        heights.append(height(slopes))

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
    return heights

def main_2b():
    """
    Theoretical
    """
    pass

def main_2c():
    raise NotImplementedError

def main(size=4, p=0.0, t_max = 1e5, seed = 0):
    """
    DEPRECATED.

    :param size: system size, the number of sites in the 1 dimensional lattice.
    :param p: probability(thresh[i]==1), 0 <= p <= 1
    :return: nothing?
    """

    slopes, thresh = relax_and_thresh_init(size, p, seed)

    ava = []
    heights = []
    scaled_height = []
    scaled_time = []
    size_sq = size * size

    for i in range(int(t_max)):
        drive(slopes)
        ava.append(relaxation(slopes, thresh, p)[1])
        h = sum(slopes)
        heights.append(h)
        scaled_height.append(h/size)
        scaled_time.append(i/(size_sq))

    print(sum(heights[2000:])/ len(heights[2000:]))

    return plt.plot(heights)
    #return plt.plot(scaled_time,scaled_height)
    #return plt.hist(heights, log=True)


if __name__ == "__main__":
    main(4, 0.5)