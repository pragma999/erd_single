from math import ceil, floor
from collections import namedtuple

""" prim_task manages primitive task and vertual server.
Intent of following types/functions are to represents task system
as like as ones expressed in paper.
In the paper based task system, C_i, T_i, D_i, represents
worst case execution time(WCET), period, and deadline.

Note that for single task, C represents just a WCET.
But for task system, C represents list of WCETs for each task.
T, D is same.
"""

# Task = namedtuple('Task', ('name', 'C', 'T', 'D'))
Server = namedtuple('VirtualServer', ('C', 'T', 'priv_task'))

class Task(object):
    def __init__(self, name, C, T, D):
        self.name = name
        self.C = C
        self.T = T
        self.D = D

    def __repr__(self):
        return "%s = (%d, %d, %d)" % (self.name, self.C, self.T, self.D)

def get_task_set(f):
    """ Retrieve C, T, D from specified file f """
    C = []
    T = []
    D = []
    names = []

    f = open(f)
    lns = f.readlines()
    i = 0
    for l in lns:
        if (l[0] == '#') or (l[0] == '\n'):
            continue

        l = l.replace(" ", "")
        names.append(l.split('=')[0])
        params = l.split('{')[1].split('}')[0].split(',')
        C.append(int(params[0]))
        T.append(int(params[1]))

        if len(params) == 2:
            D.append(int(params[1]))
        else:
            D.append(int(params[2]))

        i += 1

    ts = []
    for i, n in enumerate(names):
        ts.append(Task(names[i], C[i], T[i], D[i]))

    return ts

def rta_ts(ts):
    C, T, D = extract(ts)
    return rta(C, T)

def rta(C, T, interference=False, target=0, force_break=False):
    """ Compute the longest response time of task """
    I = [0] * len(C)
    R = [0] * len(C)
    for i, v in enumerate(C):
        R[i] = v
        while True:
            r = C[i]
            for j in range(i):
                c =  int(ceil(1.0 * R[i] / T[j]) * C[j])
                r += c
                if i <= target:
                    I[j] = c
            if R[i] == r:
                break
            if force_break and R[i] < r:
                break
            R[i] += 1
    if interference:
        return R, I
    else:
        return R

def get_interference(c, t, r):
    return int(ceil(1.0 * r / t) * c)

def mdc(Cs, Ts, Rs, Tp):
    """ Compute minimum delegation capacity """
    # print( "\t\tmdc(%d, %d, %d, %d)" % (Cs, Ts, Rs, Tp))
    phi = Ts - Cs
    n = max(0, floor(1.0 * (Tp - phi) / Ts))
    co = min(max(0, (Tp - phi) % Ts - (Rs - Cs)), Cs)
    nc = n * Cs

    v = int(nc + co)
    # print( "\t\tmdc -> phi %d, n %d, co %d, nc %d, mdc() = %d" % (phi, n, co, nc, v))
    return v

def show_task(t):
    print( "%s = { %d, %d }" % (t.name, t.C, t.T))

def task_name_to_index(ts, p):
    for i, t in enumerate(ts):
        if t.name == p:
            return i

    return -1

def extract(ts):
    """ Retrieve lists of C, T, D from taskset ts """
    C = []
    T = []
    D = []
    for i in range(len(ts)):
        C.append(ts[i].C)
        T.append(ts[i].T)
        D.append(ts[i].D)

    return C, T, D

def get_vss(ts, tau_p):
    """ Compute candidates of VS for specified task tau_p """
    if tau_p == None:
        return []

    C, T, D = extract(ts)
    R = rta(C, T)
    _VS = _get_vs(C, T, R, task_name_to_index(ts, tau_p))
    _VS.sort()
    VS = []
    vs = Server(0, 0, None)

    # ignore duplicates
    for s in _VS:
        if vs.C == s[0] and vs.T == s[1]:
            continue

        vs = Server(s[0], s[1], tau_p)
        VS.append(vs)

    return VS

def _get_vs(C, T, R, p):
    def idle(t):
        v = 0;
        for j in range(p):
            v += int(ceil(1.0 * t / T[j]) * C[j])
        return t - v

    Cs = []
    Ts = []
    VS = []

    # does tau_p finish its execution before next release of
    # higher priority tasks?
    if R[p] <= T[p - 1]:
        # Coumpute VS from Cp and Th
        for i in range(p):
            if R[p] <= T[i]:
                Cs.append(C[p])
                Ts.append(T[i])
                VS.append((C[p], T[i]))

    # Coumpute VS from idle time
    psi = T[:p]
    max_c =0
    for t in psi:
        c = idle(t)
        if c > max_c:
            max_c = c
        c = max(c, max_c)
        c = min(c, C[p])
        if c > 0:
            Cs.append(c)
            Ts.append(t)
            VS.append((c, t))

    return VS


if __name__ == "__main__":

    taskset = get_task_set('./sampleTasks/pt5.txt')
    C, T, D = extract(taskset)
    R = rta( C, T )
    VS2 = get_vss( taskset, 't2' );
    VS3 = get_vss( taskset, 't3' );
    VS4 = get_vss( taskset, 't4' );

    taskset = get_task_set('../beyond/task_set_erd_miss_20/t2591')
    C, T, D = extract(taskset)
    R = rta( C, T )

    print( taskset)
    print( C, T, D, R)

    VS2 = get_vss( taskset, 't2' );
    print( VS2)

    VS3 = get_vss( taskset, 't3' );
    print( VS3)

    VS4 = get_vss( taskset, 't4' );
    print( VS4)

    VS5 = get_vss( taskset, 't5' );
    print( VS5)

    VS6 = get_vss( taskset, 't6' );
    print( VS6)
