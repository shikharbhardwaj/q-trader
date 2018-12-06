import numpy as np
import math

# prints formatted price
def formatPrice(n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
        # vec = []
        fname = "data/" + key + ".csv"
        # lines = open(fname, "r").read().splitlines()
        # for line in lines[1:]:
                # vec.append(float(line.split(",")[4]))
        vec = np.genfromtxt(fname, skip_header=1, delimiter=',')[:, 4]
        return vec

# returns the sigmoid
def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
        d = t - n + 1
        block = None
        if d >= 0:
            block = np.array(data[d:t + 1])
        else:
            block = np.concatenate([np.array(-d * [data[0]]),
                                    np.array(data[0:t + 1])])
        return sigmoid(np.diff(block, axis=0))
