from prim_task import Task, Server, get_task_set, get_vss, task_name_to_index, rta, rta_ts, mdc, extract, get_interference
from sched import Sched, SchedErd, SchedErd2
from sched import plot
from util import lcm_list
# from exam import gen_mini_sets
import glob
import concurrent.futures

import pandas as pd
import numpy as np

import dill

MAX_N = 100000
MAX_N = 10000

enable_past_result = False
_verbose = False

def debug_print(str):
    if _verbose:
        print( str )

class TaskSystem(object):
    """ TaskSystem manages schedulability(RM, ERD, ERD2.0) of taskset.
    This class also support plotting and actual simulation.
    """
    def __init__(self, filename=None, priv_t=None, task_set=None, get_best_vs=False):
        if filename:
            self.ts = get_task_set(filename)
        else:
            self.ts = task_set
            
        self.vs = None      # best vs
        self.C, self.T, self.D = extract(self.ts)
        self.R = rta( self.C, self.T )
        self.priv_t = None
        self.pindex = 0
        self.result_sim = []

        if priv_t != None:
            for i, t in enumerate(self.ts):
                if t.name == priv_t:
                    self.priv_t = t
                    self.pindex = i

        # for ERD and ERD2.0
        vss = get_vss( self.ts, priv_t )
        self.optimal = False

        self.opt_vss = []
        # find optimal vs
        if (self.R[self.pindex] <= self.ts[self.pindex].D):
            for v in vss:
                if self.erd_1st_job_rta(v) <= self.T[0]:
                    self.optimal = True
                    opt_vs = Server(self.priv_t.C, self.priv_t.C, self.priv_t.name)
                    self.opt_vss.append(opt_vs)
                    vss = [opt_vs]
                    break

        self.vss = vss

        if get_best_vs:
            self.compute_best_vs()
        elif len(self.vss) > 0: 
            self.vs = self.vss[0]

        self.hyper_period = lcm_list(self.T)

        u = 0
        for t in self.ts:
            u += (t.C+0.0) / t.T

        self.utilization =  u
        self.mdc = self.get_mdc()

    def compute_best_vs(self, duration=10000):
        if self.priv_t == None:
            return

        if self.vss == None:
            self.vss = get_vss( self.ts, priv_t )

        rt = self.priv_t.T
        for vs in self.vss:
            wcrts, meet, dead_t = SchedErd.get_wcrt(self.ts, vs, duration)
            self.result_sim.append((vs, wcrts, meet, dead_t))
            wcrt = wcrts[self.pindex].wcrt_sim
            if (meet == True) and (wcrt <= rt):
                rt = wcrt
                self.vs = vs
        return (self.vs, rt)
    
    def get_wcrt_sim(self, test_n=0):
        return Sched.get_wcrt(self.ts, test_n)

    def get_erd_wcrt_sim(self, test_n=0):
        erds_all = []
        erds = []
        for vs in self.vss:
            # erds.append(SchedErd.get_wcrt(self.ts, vs, test_n))
            w = SchedErd.get_wcrt(self.ts, vs, test_n)
            erds_all.append(w)
            erds.append(w[0][self.pindex].wcrt_sim)

        return erds, erds_all

    def get_model(self):
        if self.priv_t == None:
            return 'RM'
        else:
            return 'ERD'

    def rm_schedulable(self):
        """ compute RTA """
        C, T, D = extract(self.ts)
        R = rta(C, T)
        for i, r in enumerate(self.R):
            if r > self.D[i]:
                return (False, self.ts[i], R)
        return (True, None, R)
    
    def erd_schedulable(self):
        sched = False
        rs = []
        for vs in self.vss:
            r = rta_erd(self.ts, self.priv_t, vs)
            if r <= self.priv_t.T:
                sched = True
                rs.append(r)
            else:
                rs.append(r)
        return sched, rs

    def get_mdc(self):
        mdcs = []
        if self.optimal:
            return [self.priv_t.C] * len(self.vss)

        for v in self.vss:
            # print( "mdc try ", v)
            tmp_ts, sv_index = self._exchange_vs(self.ts, v)
            C, T, D = extract(tmp_ts)
            R = rta(C, T)
            m = mdc(v.C, v.T, R[sv_index], self.priv_t.T)
            mdcs.append(m)

        return mdcs

    def work(self, vs, base_offset):
        ts = self.ts
        pt = self.priv_t
        chart, si, pi = get_base_chart(ts, vs, pt, base_offset)

        # print( chart )
        # print( "priv_t = (", pt.C, ", ", pt.T, "), @ srv_i = ", si, "pi ", pi)
        consumed = 0
        offset = base_offset

        for i in range(offset, len(chart)):
            if (chart[i] == si) or (chart[i] == pi):
                # print( "consumed ", chart[i], "@ ", i - offset)
                consumed += 1
                if consumed >= pt.C:
                    return i - offset + 1

        print("not meet")
        return pt.T

    def erd_1st_job_rta(self, vs):
        return self.work(vs, 0)

    def _exchange_vs(self, ts, v):
        tmp_ts = []
        sv_index = -1
        for i, t in enumerate(ts):
            if sv_index == -1:
                if v.T <= t.T:
                    tmp_ts.append(Task('vs', v.C, v.T, v.T))
                    tmp_ts.append(t)
                    sv_index = i
                else:
                    tmp_ts.append(t)

        return tmp_ts, sv_index

    def sim_schedulable(self):
        if (self.get_model() == 'ERD') and (self.result_sim == None):
            self.compute_best_vs()

        if self.vs == None:
            return False
        return True

    def plot(self, offset=0, period=32):
        if self.hyper_period < period:
            period = self.hyper_period
        plot(self.ts, self.vs, offset, period)

    def tplot(self, offset=0, period=32):
        if self.hyper_period < period:
            period = self.hyper_period
        plot(self.ts, self.vss, offset, period)

    def explot(self, offset=0, period=32):
        if self.hyper_period < period:
            period = self.hyper_period
        plot(self.ts, None, offset, period)

    def tasks(self):
        print( "U = %f" % (self.utilization))
        if self.vs != None:
            print( "VS : (%s, %s)" % (self.vs.C, self.vs.T))
        for t in self.ts:
            print( "%s : (%s, %s)" % (t.name, t.C, t.T))

    def __repr__(self):
        return "TaskSystem(ts=%s,\nvs=(%s)\nresult_sim(%s)" % (self.ts, self.vs,
                self.result_sim)

def get_base_chart(ts, vs, pt, base_offset):
    chart = [-1] * pt.T * 3
    gh = []

    vts, sv_index, p_index = get_vts(ts, vs)
    tvs_index = sv_index + 1

    for i,t in enumerate(vts):
        # if t.T < pt.T:
        if i <= p_index:
            gh.append(t)
    
    for i, t in enumerate(gh):
        if i == sv_index or i == tvs_index:
            offset = 0
        else:
            offset = base_offset
        # print "offset -> ", i, offset

        tn = i
        for j in range(offset, len(chart), t.T):
            c = 0
            r = range(j, min(len(chart), max(0, j + t.T + offset)))
            # print r
            for k in r:
                if chart[k] == -1:
                    chart[k] = tn
                    c += 1
                    if c == t.C:
                        break
    return chart, sv_index, p_index


from math import ceil, floor

def int_test(tss):
    C, T, D = extract(tss.ts)
    R, I = rta(C, T, interference=True)
    p = len(C) - 1
    for i in range(p):
        i_rm = get_interference(C[i], T[i], R[p])
        print("interference ... ", I[i])
        if I[i] != i_rm:
            print("error")

def rta_test(tss):
    rta_erd(tss.ts, tss.priv_t, tss.vs)

def rta_erd(ts, pt, vs):
    # debug_print("rta_erd try %s %s ts -> %d", % ts, pt, vs)

    if vs.C == vs.T:
        return pt.C

    vts, sv_index, p_index = get_vts(ts, vs)
    C, T, D = extract(ts)
    R_rm, I_rm = rta(C, T, interference=True, target=p_index-1)
    I_rm.insert(sv_index, 0)

    C, T, D = extract(vts)
    R_erd = rta(C[0:p_index], T[0:p_index])
    tvs_index = sv_index + 1

    I = [0] * p_index

    loop = True
    L = R_rm[p_index-1]
    while loop:
        debug_print( "RTA try L = %d" % L)
        for i, t in enumerate(vts):
            if i == sv_index:
                iv = 0
            elif i == p_index:
                break;
            elif i == tvs_index:
                iv, L_cin, cin = get_tvs_interferenct(C, T, R_erd, L, i, p_index, True)
                I_tvs = reduce_interference(C, T, R_erd, tvs_index, p_index, L, L_cin, cin, I)

                debug_print( "recuced interference %d -> %d" % (iv, I_tvs))

                I[i] = min(I_tvs, I_rm[i])
            else:
                I[i] = I_rm[i]

        # re-calc interference by updated R
        tmp_r = sum(I) + pt.C
        I_new = [0] * p_index
        for i, t in enumerate(vts):
            if i == sv_index:
                next
            elif i == p_index:
                break;
            elif i == tvs_index:
                I_new[i] = I[i]
            else :
                I_new[i] = get_interference(C[i], T[i], tmp_r)

        new_L = sum(I_new) + pt.C
        debug_print("    ... new_L %d" % new_L)
        if new_L == L:
            break
        L = new_L

        # update interference
        I_rm = I_new

    # print("I_rm  ... ", I_rm)
    # print("I     ... ", I)
    # print("I_new ... ", I_new)
    # print("R     ... ", L)

    return L
    # return sum(I) + pt.C

def get_tvs_interferenct(C, T, R_erd, L, tvs_index, p_index, is_tvs):
    # debug_print("get_tvs_interferenct ... ", C, T, R_erd, L, tvs_index, p_index, is_tvs)
    C_i = C[tvs_index]
    T_i = T[tvs_index]
    Rerd_i = R_erd[tvs_index]

    C_p = C[p_index]
    # T_p = T[p_index]

    sv_index = tvs_index - 1
    C_vs = C[sv_index]

    L_cin   = T_i - Rerd_i + C_i
    nb      = floor((L - L_cin) / T_i)
    L_body  = nb * T_i
    L_co = L - (L_cin + L_body)

    cin = min(L_cin, C_i)
    body = nb * C_i
    if is_tvs:
        co = min(max(L_co - C_vs, 0), C_i)
    else:
        co = min(L_co, C_i)

    I_tvs = cin + body + co

    # debug_print( "L ... ", L_cin, L_body, L_co, " I_tvs ... ", cin, body, co )

    return I_tvs, L_cin, cin

def reduce_interference(C, T, R_erd, i, p_index, L, L_cin, cin, I):
    T_i = T[i]
    C_i = C[i]
    Rerd_i = R_erd[i]
    C_p = C[p_index]
    T_p = T[p_index]
    C_vs = C[i-1]

    L_cod  = T_i - L_cin
    nbd    = floor((T_p - L_cod) / T_i)
    L_bd   = nbd * T_i
    L_cind = T_p - (L_cod + L_bd)

    cod = C_i - cin
    bodyd = nbd * C_i
    cind = min(max(L_cind - T_i + Rerd_i, 0), C_i)

    # debug_print( "L dash ... ", L_cind, L_bd, L_cod, " / ", cind, bodyd, cod )

    empty = max(T_p - (cind + bodyd + cod) - C_p - sum(I), 0)

    cin = max(cin - empty, 0)
    
    # debug_print( "empty ... ", empty, ", new cin ...", cin );

    L_cin = T[i] - R_erd[i] + cin
    nb      = floor((L - L_cin) / T_i)
    L_body  = nb * T_i
    L_co = L - (L_cin + L_body)

    cin = min(L_cin, C_i)
    body = nb * C_i
    dc = nb * C_vs + min(L_co, C_vs)
    if C_p <= dc:
        co = 0
    else:
        co = min(max(L_co - C_vs, 0), C_i)

    I_tvs = cin + body + co

    # debug_print( "L ... ", L_cin, L_body, L_co, " I_tvs ... ", cin, body, co )

    return I_tvs

def get_vts(ts, v):
    tmp_ts = []
    added = False;
    sv_index = -1
    for i, t in enumerate(ts):
        if (added == False) and (v.T <= t.T):
            tmp_ts.append(Task('vs', v.C, v.T, v.T))
            tmp_ts.append(t)
            sv_index = i
            added = True
        else:
            tmp_ts.append(t)

    p_index = task_name_to_index(tmp_ts, v.priv_task)
    return tmp_ts, sv_index, p_index

matrix = []

def get_matrix():
    return matrix

def test(ts, priv_t, test_name="", rta_check=False):
    """ test() function performes all of evaluation that including
    RM, DM, ERD response time/jitter mesurement by simulation and RTA
    """
    print( "test # of task %d U=%f" % (ts.pindex, ts.utilization), flush=True)
    dm_opt = False

    # RM simulation
    rs = ts.get_wcrt_sim(test_n=0)
    rm_rt = ts.R[ts.pindex]
    rm_afj = rs[0][ts.pindex].afj
    rm_rfj = rs[0][ts.pindex].rfj

    # DM simulation
    pt = ts.ts[ts.pindex]
    dm_rt = rm_rt
    dm_afj = rm_afj
    dm_rfj = rm_rfj 
    for i, hp in enumerate(ts.T):
        # shorten deadline one by one
        ts.ts[ts.pindex] = Task(pt.name, pt.C, pt.T, hp-1)
        rs = ts.get_wcrt_sim(test_n=0)
        if rs.meet:
            if i == 0:
                dm_opt = True
            dm_rt  = rs[0][ts.pindex].wcrt_sim
            dm_afj = rs[0][ts.pindex].afj
            dm_rfj = rs[0][ts.pindex].rfj
            break

    # ERD simulation
    rta_anomaly = False
    if dm_opt:
        erd_afj = dm_afj
        erd_rfj = dm_rfj
        erd_sched  = True
        r_erd  = [ts.priv_t.C]
        sim_wcrts = [ts.priv_t.C]
    elif ts.vss==[]:
        erd_afj = dm_afj
        erd_rfj = dm_rfj
        erd_sched  = True
        r_erd  = [ts.R[ts.pindex]]
        sim_wcrts = [ts.R[ts.pindex]]
    else:
        ts.ts[ts.pindex] = Task(pt.name, pt.C, pt.T, pt.T)
        ws, wsall = ts.get_erd_wcrt_sim(test_n=0)
        sim_wcrts = []
        for i, w in enumerate(ws):
            # appends wcrt of target task
            # sim_wcrts.append(w[0][ts.pindex].wcrt_sim)
            sim_wcrts.append(w)

        erd_afj = dm_afj
        erd_rfj = dm_rfj
        for i, tm in enumerate(wsall):
            erd_afj = min(erd_afj, tm[0][ts.pindex].afj)
            erd_rfj = min(erd_rfj, tm[0][ts.pindex].rfj)

        # ERD schedulable
        erd_sched, r_erd = ts.erd_schedulable()

        diff = np.array(sim_wcrts) - np.array(r_erd)
        if max(diff) > 0:
            rta_anomaly = True
            print( "\t!!!!!! anomaly !!!!!!" )

    r_erd.append(dm_rt)
    sim_wcrts.append(dm_rt)
    speed = min(r_erd) - rm_rt
    if priv_t.C == min(r_erd):
        opt = True
    else:
        opt = False

    if rta_check:
        rm_schedulable = ts.rm_schedulable()
    else:
        rm_schedulable = True

    matrix.append(( \
        test_name, priv_t.name, ts.utilization, rm_schedulable, erd_sched,  \
        rm_rt, speed, dm_rt, sim_wcrts, r_erd,    \
        [rm_afj, dm_afj, erd_afj], [rm_rfj, dm_rfj, erd_rfj], rta_anomaly, opt))
    return ts


def simple_test(filename):
    # RM style task system ( w/o server )
    ts = TaskSystem(filename=filename, priv_t=None)

    if (len(ts.ts) == 0):
        return

    print( "test %s with %s tasks" % (filename, ts.ts[-1].name))

    # RTA without server
    rm_schedulable, miss_t, R = ts.rm_schedulable()

    if miss_t == None:
        # assume final task is target task
        priv_t = ts.ts[-1]
    else:
        for i, t in enumerate(ts.ts):
            if t.name == miss_t.name:
                break

        # restriction : only one missed task is allowed
        if len(ts.ts[i:]) != 1:
            print( "\t%s has several missed tasks!\n" % (filename))
            return
        priv_t = miss_t

    # Task set with server
    ts = TaskSystem(filename=filename, priv_t=priv_t.name)
    if ts.vss == []:
        print( "\t%s has no room to execute priv task!\n" % (filename))
        return

    return test(ts, priv_t, rta_check=True)

 
# test with exist skip
def test_by_file(filename, mid=False):
    # RM style task system ( w/o server )
    filename = filename.replace('\\', '/')

    print( "\ntest_by_file %s" % (filename))
    ts = TaskSystem(filename=filename, priv_t=None)

    if (len(ts.ts) == 0):
        return

    print( "%s tasks" % (ts.ts[-1].name))

    # RTA without server
    rm_schedulable, miss_t, R = ts.rm_schedulable()

    if miss_t == None:
        if mid:
            priv_t = ts.ts[int(len(ts.ts)/2)]
        else:
            # assume final task is target task
            priv_t = ts.ts[-1]
    else:
        for i, t in enumerate(ts.ts):
            if t.name == miss_t.name:
                break

        # restriction : only one missed task is allowed
        if len(ts.ts[i:]) != 1:
            print( "\t%s has several missed tasks!\n" % (filename))
            return
        priv_t = miss_t

    # Task set with server
    ts = TaskSystem(filename=filename, priv_t=priv_t.name)
    if ts.vss == []:
        print( "\t%s has no room to execute priv task!\n" % (filename))
        return

    return test(ts, priv_t)

def test_schedcat(tss, mid=False):
    # RM style task system ( w/o server )
    filename = filename.replace('\\', '/')

    print( "\ntest_by_file %s" % (filename))
    ts = TaskSystem(filename=filename, priv_t=None)

    if (len(ts.ts) == 0):
        return

    print( "%s tasks" % (ts.ts[-1].name))

    # RTA without server
    rm_schedulable, miss_t, R = ts.rm_schedulable()

    if miss_t == None:
        if mid:
            priv_t = ts.ts[int(len(ts.ts)/2)]
        else:
            # assume final task is target task
            priv_t = ts.ts[-1]
    else:
        for i, t in enumerate(ts.ts):
            if t.name == miss_t.name:
                break

        # restriction : only one missed task is allowed
        if len(ts.ts[i:]) != 1:
            print( "\t%s has several missed tasks!\n" % (filename))
            return
        priv_t = miss_t

    # Task set with server
    ts = TaskSystem(filename=filename, priv_t=priv_t.name)
    if ts.vss == []:
        print( "\t%s has no room to execute priv task!\n" % (filename))
        return

    # ERD schedulable
    erd_sched, r_erd = ts.erd_schedulable()

    # sim result
    ok = False
    if enable_past_result:
        res = df_all[df_all['file'] == filename]
        if len(res) == 1:
            l = res['sim']
            ll = l[l.index[0]]
            sim_wcrts = [int(x.strip()) for x in ll.split(',')]
            if len(sim_wcrts) == len(r_erd):
                ok = True

    if ok == False:
        # ERD simulation
        ws, _ = ts.get_erd_wcrt_sim(test_n=0)
        sim_wcrts = []
        for i, w in enumerate(ws):
            # appends wcrt of target task
            # sim_wcrts.append(w[0][ts.pindex].wcrt_sim)
            sim_wcrts.append(w)

    diff = np.array(sim_wcrts) - np.array(r_erd)
    if max(diff) > 0:
        rta_anomaly = True
        print( "\t!!!!!! anomaly !!!!!!" )
    else:
        rta_anomaly = False

    r_rm = R[ts.pindex]
    val = min(r_erd) - r_rm
    if priv_t.C == min(r_erd):
        opt = True
    else:
        opt = False

    matrix.append((filename, priv_t.name, ts.utilization, rm_schedulable, erd_sched, r_rm, val, sim_wcrts, r_erd, rta_anomaly, opt))
    return ts

def get_df():
    return pd.DataFrame(matrix, columns=['file', 'priv', 'U', 'RM_sched', 'ERD_sched', 'RM', 'speed', 'DM', 'sim', 'RTA', 'AFJ', 'RFJ', 'anomaly', 'opt'])

def show_matrix():
    for m in matrix:
        sim_wcrts = np.array(m[3])
        rtas = np.array(m[4])
        flag = False
        for v in (sim_wcrts - rtas):
            if v > 0:
                flag = True
        if flag:
            print( m[0], sim_wcrts - rtas )

def check_anomaly(df):
    return df[df.anomaly==True]

if __name__ == "__main__":
    pd.set_option("display.width", 120)

    mini = TaskSystem(filename='./sampleTasks/minisp.txt', priv_t='t3')
    pt5 = TaskSystem(filename='./sampleTasks/pt5.txt', priv_t='t4')

    if False:
        tss = gen_mini_sets()
        for ts in tss:
            for t in ts:
                test(t, t.ts[-1])

    if True:
        mini = TaskSystem(filename='./sampleTasks/mini.txt', priv_t='t2')
        dm   = TaskSystem(filename='./sampleTasks/dm.txt')
        cata = TaskSystem(filename='./sampleTasks/cata.txt', priv_t='t4')
        pt2 = TaskSystem(filename='./sampleTasks/pt2.txt', priv_t='t4')
        pt3 = TaskSystem(filename='./sampleTasks/pt3.txt', priv_t='t3')
        pt4 = TaskSystem(filename='./sampleTasks/pt4.txt', priv_t='t3')
        pt5 = TaskSystem(filename='./sampleTasks/pt5.txt', priv_t='t4')
        pt6 = TaskSystem(filename='./sampleTasks/pt6.txt', priv_t='t4')
        pt7 = TaskSystem(filename='./sampleTasks/pt7.txt', priv_t='t4')
        d1 = TaskSystem(filename='./sampleTasks/d1.txt', priv_t='t4')
        imp = TaskSystem(filename='./sampleTasks/imp.txt', priv_t='t3')

        simple_test('./sampleTasks/mini.txt')
        simple_test('./sampleTasks/pt7.txt')
        simple_test('./sampleTasks/pt4.txt')
        simple_test('./sampleTasks/pt5.txt')
        simple_test('../beyond/task_set_erd_miss_50/t1392')

    if False:
        with open('failed_ts', 'rb') as f:
            missed_fs = dill.load(f)

        fss = missed_fs.file

        for f in fss:
            simple_test( f )
        
        df = pd.DataFrame(matrix, columns=['file', 'priv', 'rta', 'rm', 'erd', 'erd2', 'ts', 'wcrt'])
        df1 = df[df['erd'] == False]
        df2 = df1[df1['erd2'] == True]

        df3 = df[df['erd'] == True]
        df4 = df3[df3['erd2'] == False]

    df = get_df()
