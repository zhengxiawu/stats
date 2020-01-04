import numpy as np
import fractions
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


class CategoricalOneDim(object):
    def __init__(self, random_variable, prob = None, is_random=False, random_level=0.1):
        self.random_variable = np.array(random_variable)
        if is_random:
            self.prob = np.random.normal(1./len(self.random_variable), random_level, len(self.random_variable))
            self.prob = np.array([fractions.Fraction(i) for i in self.prob])
            self.prob[self.prob < 0] = 0
            self.prob = self.prob / np.sum(self.prob)
        else:
            if prob is not None:
                self.prob = prob
            else:
                self.prob = [fractions.Fraction(1, len(self.random_variable))] * len(self.random_variable)
        self.prob = np.array(self.prob)


def add_category_distribution(a, b):
    z_random_variable = []
    z_prob = []
    for i in range(len(a.random_variable)):
        for j in range(len(b.random_variable)):
            _z_var = a.random_variable[i] + b.random_variable[j]
            _z_prob = a.prob[i] * b.prob[j]
            if _z_var in z_random_variable:
                _z_var_index = z_random_variable.index(_z_var)
                z_prob[_z_var_index] += _z_prob
            else:
                z_random_variable.append(_z_var)
                z_prob.append(_z_prob)
    z_random_variable = np.array(z_random_variable)
    z_prob = np.array(z_prob)
    _z_var_index_sort = np.argsort(z_random_variable)
    z_random_variable = z_random_variable[_z_var_index_sort]
    z_prob = z_prob[_z_var_index_sort]
    return CategoricalOneDim(z_random_variable, z_prob)


def get_prob(N, M):
    init_random_list = np.array(list(range(M))) + 1
    is_random = False
    former_dist = CategoricalOneDim(init_random_list, is_random=is_random)
    new_dist = None
    total = []
    for i in range(N):
        if new_dist:
            new_dist = add_category_distribution(new_dist, former_dist)
        else:
            new_dist = CategoricalOneDim(init_random_list, is_random=is_random)
        # a = [int(i) for i in new_dist.prob * (M**(i+1))]
        # print(a)
        # _ = []
        # #
        # new_i = i
        # for k in range(len(a)):
        #     sub_list = []
        #     for j in range(math.floor(k/M)+1):
        #         sub_list.append(((-1)**j)*nCr(new_i+1, j)*nCr(new_i+1+k-1-M*j, new_i))
        #
        #     _.append(sum(sub_list))
        # print([int(m) for m in _])
        # neg_dist_2 = copy.deepcopy(new_dist)
        # neg_dist_2.random_variable *= -1
        # neg_dist_2 = add_category_distribution(new_dist, neg_dist_2)
        # random_variable = np.array(neg_dist_2.random_variable)
        # a = [int(i) for i in neg_dist_2.prob * ((M ** (i + 1)) ** 2)]
        # testa = np.array(a)
        # testa = testa[(random_variable >= 1 - M) & (random_variable <= M - 1)]
        # test = np.sum(testa)
        # total.append(test)
        # # print(test)
        # print(a)
    neg_dist = copy.deepcopy(new_dist)
    neg_dist.random_variable *= -1
    new_dist = add_category_distribution(new_dist, neg_dist)

    prob = 0
    this_M = 1
    number_list = np.flip(np.array(range(this_M))+1)
    for i in range(len(new_dist.random_variable)):
        # if -1 * M < new_dist.random_variable[i] < M:
        #     prob += float(new_dist.prob[i])
        if 0 <= new_dist.random_variable[i] < this_M:
            prob += float(new_dist.prob[i])
        elif -1*this_M < new_dist.random_variable[i] < 0:
            total_number = np.sum(number_list[:int(-1*new_dist.random_variable[i])]) + (this_M*(this_M-1)/2.)
            prob += float(new_dist.prob[i]) * (this_M*(this_M-1)/2.)/total_number
        else:
            prob += float(new_dist.prob[i]) * (this_M-1)/(2.*this_M)
    return prob


if __name__ == '__main__':

    N = 16
    M = 9
    get_prob(N, M)
    X = np.arange(2, N+2, 1)
    Y = np.arange(2, M+2, 1)
    X, Y = np.meshgrid(X, Y)
    prob = np.zeros([M, N])
    for i in range(2, N+2):
        for j in range(2, M+2):
            prob[j-2, i-2] = get_prob(i, j)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, prob, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    plt.show()
    pass

