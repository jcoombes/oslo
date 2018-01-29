import pytest
import oslo
import numpy as np

"""
Unit Tests
"""

if __name__ == "__main__":
    pytest.main()

@pytest.mark.finished
def test_drive():
    assert oslo.drive([0, 0, 0]) == [1, 0, 0]
    assert oslo.drive([-1, 0, 0]) == [0, 0, 0]
    assert oslo.drive([54364, 0, 0]) == [54365, 0, 0]
    assert oslo.drive([0]) == [1]
    assert np.array_equal(oslo.drive(np.array([25, 12, 7])), np.array([26, 12, 7]))

    a = [3, 0, 0]
    oslo.drive(a)
    assert a == [4, 0, 0]

    oslo.drive(a)
    assert a == [5, 0, 0]

    oslo.drive(a)
    assert a == [6, 0, 0]

    with pytest.raises(ValueError):
        oslo.drive([])
        oslo.drive(np.array([]))

@pytest.mark.finished
def test_relax():
    assert oslo.relax(0, [3, 2, 1]) == [1, 3, 1]
    assert oslo.relax(1, [3, 2, 1]) == [4, 0, 2]
    assert oslo.relax(2, [3, 2, 1]) == [3, 3, 0]

    a = [11, 0, 0, 0]
    oslo.relax(0, a)
    assert a == [9, 1, 0, 0] #These test results have been pre-computed
    oslo.relax(0, a)
    assert a == [7, 2, 0, 0] #Also, let's assume every threshhold == 1
    oslo.relax(0, a)
    assert a == [5, 3, 0, 0] #Hm, notice how this is like a tree.
    oslo.relax(0, a)
    assert a == [3, 4, 0, 0] # In order to relax from initial state
    oslo.relax(0, a)
    assert a == [1, 5, 0, 0] # To final state, you don't need to check
    oslo.relax(1, a)
    assert a == [2, 3, 1, 0] # z > z_th using a nested while for loop
    oslo.relax(0, a)
    assert a == [0, 4, 1, 0] # This is O(n^2). Can we be smarter.
    oslo.relax(1, a)
    assert a == [1, 2, 2, 0]
    oslo.relax(1, a)
    assert a == [2, 0, 3, 0] #Also, when running relax(n, slopes) m times,
    oslo.relax(0, a)
    assert a == [0, 1, 3, 0] #we can just subtract 2m from z, add m to z[n+1]
    oslo.relax(2, a)
    assert a == [0, 2, 1, 1]
    oslo.relax(1, a)
    assert a == [1, 0, 2, 1]
    oslo.relax(2, a)
    assert a == [1, 1, 0, 2]
    oslo.relax(3, a)
    assert a == [1, 1, 1, 1]

    b = [2, 2, 2, 2]
    oslo.relax(0, b) #This hints at another 'inverse' representation.
    oslo.relax(1, b)
    oslo.relax(2, b)
    oslo.relax(3, b)

    assert b == [1, 2, 2, 2]

    with pytest.raises(ValueError):
        oslo.relax(0, [])

    with pytest.raises(IndexError):
        oslo.relax(7, [6, 5, 4, 3, 2, 1])
        oslo.relax(6, [6, 5, 4, 3, 2, 1])
        oslo.relax(-1,[6, 5, 4, 3, 2, 1])

@pytest.mark.finished
@pytest.mark.slow
def test_relaxation():
    #When p=1, this simplifies to 1 dimensional BTW model.
    assert oslo.relaxation([11,0,0,0],[1,1,1,1], p=1)[0] == [1,1,1,1]
    assert oslo.relaxation([0,0],[1,1], p=1)[0] == [0,0]
    assert oslo.relaxation([1,2],[1,1], p=1)[0] == [1,2]
    assert oslo.relaxation([1,1,1,1],[1,1,1,1],p=1)[0] == [1,1,1,1]

    assert oslo.relaxation([222, 0,0,0],[2,2,2,2],p=1)[0] == [1,1,1,1] #initial threshholds get smoothed out.

    assert oslo.relaxation([11,0,0,0],[1,1,1,1],p=1)[1] == 14 #Correct number of avalanches? See test_drive for proof.
    assert oslo.relaxation([111, 0, 0, 0], [1, 1, 1, 1], p=1)[1] >= 10 * 14 # 414
    assert oslo.relaxation([1111,0,0,0],[1,1,1,1],p=1)[1] >= 10 * 140 #4414
    assert oslo.relaxation([10000, 0, 0, 0], [1, 1, 1, 1], p=1)[1] >= 10 * 140  # 39970.
    assert oslo.relaxation([11111,0,0,0],[1,1,1,1],p=1)[1] >= 10 * 1400 #44414?

    #when p=0, this simplifies to a variant of the 1-dimensional BTW model.
    assert oslo.relaxation([1,1,1,1],[2,2,2,2],p=0)[0] == [1,1,1,1]
    assert oslo.relaxation([22,0,0,0],[2,2,2,2],p=0)[0] == [2,2,2,2]
    assert oslo.relaxation([222,0,0,0],[2,2,2,2],p=0)[0] == [2,2,2,2]
    assert oslo.relaxation([2, 3], [1, 2], p=0)[0] == [2,2]

    with pytest.raises(IndexError):
        oslo.relaxation([1,1,1,1],[2], p=0)[1] # If len(thresh) < len(slopes), IndexError
        oslo.relaxation([2, 3], [1, 2, 1, 2], p=0)[1]

@pytest.mark.finished
def test_thresh_update():
    assert oslo.thresh_update(1, [1,1,1,1],p=0) == [1,2,1,1]
    assert oslo.thresh_update(2, [1,'dog',3],p=0) == [1,'dog',2]
    assert oslo.thresh_update(1, [1,'dog',3],p=0) == [1,2,3]
    assert oslo.thresh_update(10,[2]*10 + [2], p=1) == [2,2,2,2,2,2,2,2,2,2,1]

    with pytest.raises(IndexError):
        oslo.thresh_update(5,[1,1,1,1,1],p=0)

    with pytest.raises(ValueError):
        oslo.thresh_update(5, [2,'t',54,5,6,8],p=5000)

@pytest.mark.finished
@pytest.mark.slow
def test_thresh_is_random():
    np.random.seed(0)
    sum = 0
    sample = 1e4
    for i in np.arange(sample):
        sum += oslo.thresh_update(14,[1,]*30,p=0.5)[14]
    assert sum/sample >= 1.49
    assert sum/sample <= 1.51

    sum = 0
    sample = 1e4
    for i in np.arange(sample):
        sum += oslo.thresh_update(14, [1, ] * 30, p=0.1)[14]
    assert sum / sample >= 1.89
    assert sum / sample <= 1.91


    sum = 0
    sample = 1e4
    for i in np.arange(sample):
        sum += oslo.thresh_update(14, [1, ] * 30, p=0.75)[14]
    assert sum / sample >= 1.24
    assert sum / sample <= 1.26

def test_main():
    assert 1