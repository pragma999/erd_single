"This source code is based on schedcat  https://github.com/brandenburg/schedcat.git"

from __future__ import division

from math import trunc, exp, log

import random

from prim_task import Task, extract, rta
from tasks import TaskSystem
import concurrent.futures

def uniform_int(minval, maxval):
    "Create a function that draws ints uniformly from {minval, ..., maxval}"
    def _draw():
        return random.randint(minval, maxval)
    return _draw

def uniform(minval, maxval):
    "Create a function that draws floats uniformly from [minval, maxval]"
    def _draw():
        return random.uniform(minval, maxval)
    return _draw

def log_uniform(minval, maxval):
    "Create a function that draws floats log-uniformly from [minval, maxval]"
    def _draw():
        return exp(random.uniform(log(minval), log(maxval))) 
    return _draw

def log_uniform_int(minval, maxval):
    "Create a function that draws ints log-uniformly from {minval, ..., maxval}"
    draw_float = log_uniform(minval, maxval + 1)
    def _draw():
        val = int(draw_float())
        val = max(minval, val)
        val = min(maxval, val)
        return val
    return _draw

def uniform_choice(choices):
    "Create a function that draws uniformly elements from choices"
    selector = uniform_int(0, len(choices) - 1)
    def _draw():
        return choices[selector()]
    return _draw

def truncate(minval, maxval):
    def _limit(fun):
        def _f(*args, **kargs):
            val = fun(*args, **kargs)
            return min(maxval, max(minval, val))
        return _f
    return _limit

def redraw(minval, maxval):
    def _redraw(dist):
        def _f(*args, **kargs):
            in_range = False
            while not in_range:
                val = dist(*args, **kargs)
                in_range = minval <= val <= maxval
            return val
        return _f
    return _redraw

def exponential(minval, maxval, mean, limiter=redraw):
    """Create a function that draws floats from an exponential
    distribution with expected value 'mean'. If a drawn value is less
    than minval or greater than maxval, then either another value is
    drawn (if limiter=redraw) or the drawn value is set to minval or
    maxval (if limiter=truncate)."""
    def _draw():
        return random.expovariate(1.0 / mean)
    return limiter(minval, maxval)(_draw)

def multimodal(weighted_distributions):
    """Create a function that draws values from several distributions
    with probability according to the given weights in a list of
    (distribution, weight) pairs."""
    total_weight = sum([w for (d, w) in weighted_distributions])
    selector = uniform(0, total_weight)
    def _draw():
        x = selector()
        wsum = 0
        for (d, w) in weighted_distributions:
            wsum += w
            if wsum >= x:
                return d()
        assert False # should never drop off
    return _draw



class TaskGenerator(object):
    """Sporadic task generator"""

    def __init__(self, period, util, deadline=lambda x, y: y):
        """Creates TaskGenerator based on a given a period and
        utilization distributions."""
        self.period    = period
        self.util      = util
        self.deadline  = deadline

    def tasks(self, max_tasks=None, max_util=None, squeeze=False,
              time_conversion=trunc):
        """Generate a sequence of tasks until either max_tasks is reached
        or max_util is reached. If max_util would be exceeded and squeeze is
        true, then the last-generated task's utilization is scaled to exactly
        match max_util. Otherwise, the last-generated task is discarded.
        time_conversion is used to convert the generated (non-integral) values
        into integral task parameters.
        """
        count = 0
        usum  = 0
        task_set = []
        while ((max_tasks is None or count < max_tasks) and
               (max_util is None  or usum  < max_util)):
            period   = self.period()
            util     = self.util()
            cost     = period * util
            deadline = self.deadline(cost, period)
            # scale as required
            period   = max(1,    int(time_conversion(period)))
            cost     = max(1,    int(time_conversion(cost)))
            deadline = max(1, int(time_conversion(deadline)))
            util = cost / period
            count  += 1
            usum   += util
            if max_util and usum > max_util:
                if squeeze:
                    # make last task fit exactly
                    util -= (usum - max_util)
                    cost = max(trunc(period * util), 1)
                else:
                    break
            task_set.append((period, cost, period, deadline))
            # yield ts('t0', cost, period, deadline)
        return task_set

    def make_task_set(self, *extra, **kextra):
        return ts.TaskSystem(self.tasks(*extra, **kextra))

def get_task_set(tg, util, cn, num):
    rt  = 0
    while True:
        ts = tg.tasks(max_util=util, squeeze=True)
        if len(ts) == 1:
            continue

        ts.sort()
        tasks = []

        u = 0.0
        for i, t in enumerate(ts):
            tn = 't'+str(i+1)
            tasks.append(Task(tn, t[1], t[2], t[3]))
            u += t[1] / t[2]

        if (u < util) or (u > util + 0.019) or (u > 1.0):
            continue

        C, T, D = extract(tasks)
        R = rta( C, T )

        if R[-1] > T[-1]:
            rt  += 1 
            if rt % 500 == 0:
                print("# ", num, " retry ", rt, cn)
            continue

        # print( u, tasks)
        t = TaskSystem(task_set=tasks, priv_t=tn)

        if t.vss == []:
            continue

        # print( t.rm_schedulable(), t.utilization, t.ts )
        if t.rm_schedulable()[0]:
            return t

def test(n):
    sleep(1)
    print( n )

def get_task_sets(n, util):
    ts_set = []

    # tg = TaskGenerator(uniform_int(10, 100), uniform(0.1, 0.4))
    # tg = TaskGenerator(uniform_int(10, 100), exponential(0, 1, 0.25))
    tg = TaskGenerator(uniform_int(10, 100), exponential(0, 1, 0.10))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while len(ts_set) < n:
            f1 = executor.submit(get_task_set, tg, util, len(ts_set), 1)
            f2 = executor.submit(get_task_set, tg, util, len(ts_set), 2)
            f3 = executor.submit(get_task_set, tg, util, len(ts_set), 3)
            f4 = executor.submit(get_task_set, tg, util, len(ts_set), 4)
            ts_set.append(f1.result())
            ts_set.append(f2.result())
            ts_set.append(f3.result())
            ts_set.append(f4.result())

    return ts_set

if __name__ == "__main__":
    tss = get_task_sets(10, 0.97)
