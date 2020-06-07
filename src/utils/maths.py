import math
import logging

import numpy as np


logger = logging.getLogger(__name__)


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def running_mean(x, n_points):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n_points:] - cumsum[:-n_points]) / float(n_points)


def get_avg_n_points(list_points, n_points):
    n_moving_points = int(np.ceil(len(list_points) / n_points))
    y_all = running_mean(list_points, n_moving_points)
    y = [list_points[0]] + [j for i, j in enumerate(y_all) if i % n_moving_points == 0] + [y_all[-1]]
    return y


def get_min_n_points(list_points, n_points):
    n_moving_points = int(np.ceil(len(list_points) / n_points))
    min_n_points = []
    for i_point_batch in range(n_points):
        point_batch = list_points[i_point_batch*n_moving_points:(i_point_batch+1)*n_moving_points]
        min_n_points.append(min(point_batch))
    return [list_points[0]] + min_n_points + [list_points[-1]]


def get_max_n_points(list_points, n_points):
    n_moving_points = int(np.ceil(len(list_points) / n_points))
    min_n_points = []
    for i_point_batch in range(n_points):
        point_batch = list_points[i_point_batch*n_moving_points:(i_point_batch+1)*n_moving_points]
        min_n_points.append(max(point_batch))
    return [list_points[0]] + min_n_points + [list_points[-1]]
