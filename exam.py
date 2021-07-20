import pandas as pd
import numpy as np
import os
from gen import get_task_sets
from tasks import TaskSystem
from matplotlib import pyplot
from tasks import test, get_df
import dill
import concurrent.futures

import time

TEST_N = 1000

show_plt = True
one_p = False
five_p = False
five_p = True

test_mini = False
test_mini = True

def trend(label, df):
    ntotal = len(df)
    
    avg_rm = np.average(df.rm)
    # avg shorten (regardless whether ERD has a value)
    avg_sim = np.average(df.sim / df.rm)
    avg_rta = np.average(df.rta / df.rm)
    avg_dm = np.average(df.dm / df.rm)

    if ntotal >  0:
        # of shorten
        nsim = sum(df.sim < df.rm) / ntotal
        nrta = sum(df.rta < df.rm) / ntotal
        ndm  = sum(df.dm < df.rm) / ntotal
    else:
        nsim = 0
        nrta = 0
        ndm  = 0

    # data frame which succeeded to shorten response time compared with RM's RTA
    dfs = df[df.sim < df.rm]
    dfr = df[df.rta < df.rm]
    dfd  = df[df.dm < df.rm]

    # succeeded ratio
    navg_sim = np.average(dfs.sim / dfs.rm)
    navg_rta = np.average(dfr.rta / dfr.rm)
    navg_dm = np.average(dfd.dm / dfd.rm)

    # rfj
    navn_rfj_rm = np.average(df.rm_rfj)
    navn_rfj_dm = np.average(df.dm_rfj)
    navn_rfj_erd = np.average(df.erd_rfj)
    navn_afj_rm = np.average(df.rm_afj)
    navn_afj_dm = np.average(df.dm_afj)
    navn_afj_erd = np.average(df.erd_afj)

    return [label, avg_rm, avg_sim, avg_rta, avg_dm, nsim, nrta, ndm, navg_sim, navg_rta, navg_dm,  \
            navn_rfj_rm, navn_rfj_dm, navn_rfj_erd, navn_afj_rm, navn_afj_dm, navn_afj_erd ]

def construct_df(f):
    df = pd.read_csv(f)

    # sim 
    l = []
    for v in df.sim:
        x = v.replace('[', '')
        x = x.replace(']', '')
        l.append(min(list(map(lambda v: int(v), list(x.split(','))))))

    df['sim'] = l

    l = []
    for i, v in enumerate(df.RTA):
        x = v.replace('[', '')
        x = x.replace(']', '')
        l.append(min(list(map(lambda v: int(v), list(x.split(','))))))

    df['rta'] = l

    # DM
    df['dm'] = df['DM']
    df['rm'] = df['RM']

    # jitter
    lrm = []
    ldm = []
    lerd = []
    for v in df.AFJ:
        x = v.replace('[', '')
        x = x.replace(']', '')
        vl = list(map(lambda v: int(v), list(x.split(','))))
        lrm.append(vl[0])
        ldm.append(vl[1])
        lerd.append(vl[2])

    df['rm_afj'] = lrm
    df['dm_afj'] = ldm
    df['erd_afj'] = lerd

    lrm = []
    ldm = []
    lerd = []
    for v in df.RFJ:
        x = v.replace('[', '')
        x = x.replace(']', '')
        vl = list(map(lambda v: int(v), list(x.split(','))))
        lrm.append(vl[0])
        ldm.append(vl[1])
        lerd.append(vl[2])

    df['rm_rfj'] = lrm
    df['dm_rfj'] = ldm
    df['erd_rfj'] = lerd


    df = df[df.RM_sched==True]
    df = df.drop('file', axis=1)
    df = df.drop('anomaly', axis=1)
    df = df.drop('ERD_sched', axis=1)
    df = df.drop('RM_sched', axis=1)
    df = df.drop('RM', axis=1)
    df = df.drop('DM', axis=1)
    df = df.drop('RTA', axis=1)
    df = df.drop('AFJ', axis=1)
    df = df.drop('RFJ', axis=1)
    df = df.drop('opt', axis=1)
    df = df.drop('speed', axis=1)


    tmp = df[df.U >= 0.60]
    df60 = tmp[tmp.U < 0.62]

    tmp = df[df.U >= 0.62]
    df62 = tmp[tmp.U < 0.64]

    tmp = df[df.U >= 0.64]
    df64 = tmp[tmp.U < 0.66]

    tmp = df[df.U >= 0.66]
    df66 = tmp[tmp.U < 0.68]

    tmp = df[df.U >= 0.68]
    df68 = tmp[tmp.U < 0.70]

    tmp = df[df.U >= 0.70]
    df70 = tmp[tmp.U < 0.72]

    tmp = df[df.U >= 0.72]
    df72 = tmp[tmp.U < 0.74]

    tmp = df[df.U >= 0.74]
    df74 = tmp[tmp.U < 0.76]

    tmp = df[df.U >= 0.76]
    df76 = tmp[tmp.U < 0.78]

    tmp = df[df.U >= 0.78]
    df78 = tmp[tmp.U < 0.80]

    tmp = df[df.U >= 0.80]
    df80 = tmp[tmp.U < 0.82]

    tmp = df[df.U >= 0.82]
    df82 = tmp[tmp.U < 0.84]

    tmp = df[df.U >= 0.84]
    df84 = tmp[tmp.U < 0.86]

    tmp = df[df.U >= 0.86]
    df86 = tmp[tmp.U < 0.88]

    tmp = df[df.U >= 0.88]
    df88 = tmp[tmp.U < 0.90]

    tmp = df[df.U >= 0.90]
    df90 = tmp[tmp.U < 0.92]

    tmp = df[df.U >= 0.92]
    df92 = tmp[tmp.U < 0.94]

    tmp = df[df.U >= 0.94]
    df94 = tmp[tmp.U < 0.96]

    tmp = df[df.U >= 0.96]
    df96 = tmp[tmp.U < 0.98]

    df98 = df[df.U >= 0.98]

    dfs = [
    # ('62', df60),
    # ('64', df62),
    # ('66', df64),
    # ('68', df66),
    # ('70', df60),
    ('72', df70),
    ('74', df72),
    ('76', df74),
    ('78', df76),
    ('80', df78),
    ('82', df80),
    ('84', df82),
    ('86', df84),
    ('88', df86),
    ('90', df88),
    ('92', df90),
    ('94', df92),
    ('96%', df94)
    ]

    tmp = df[df.U >= 0.70]
    ddf70 = tmp[tmp.U < 0.75]

    tmp = df[df.U >= 0.75]
    ddf75 = tmp[tmp.U < 0.80]

    tmp = df[df.U >= 0.80]
    ddf80 = tmp[tmp.U < 0.85]

    tmp = df[df.U >= 0.85]
    ddf85 = tmp[tmp.U < 0.90]

    tmp = df[df.U >= 0.90]
    ddf90 = tmp[tmp.U < 0.95]

    ddf95 = df[df.U >= 0.95]

    fdfs = [
    ('70', ddf70),
    ('75', ddf75),
    ('80', ddf80),
    ('85', ddf85),
    ('90', ddf90),
    ('95%', ddf90)
    ]

    tmp =   df[df.U >= 0.70]
    df70 = tmp[tmp.U < 0.71]
    tmp =   df[df.U >= 0.71]
    df71 = tmp[tmp.U < 0.72]
    tmp =   df[df.U >= 0.72]
    df72 = tmp[tmp.U < 0.73]
    tmp =   df[df.U >= 0.73]
    df73 = tmp[tmp.U < 0.74]
    tmp =   df[df.U >= 0.74]
    df74 = tmp[tmp.U < 0.75]
    tmp =   df[df.U >= 0.75]
    df75 = tmp[tmp.U < 0.76]
    tmp =   df[df.U >= 0.76]
    df76 = tmp[tmp.U < 0.77]
    tmp =   df[df.U >= 0.77]
    df77 = tmp[tmp.U < 0.77]
    tmp =   df[df.U >= 0.78]
    df78 = tmp[tmp.U < 0.79]
    tmp =   df[df.U >= 0.79]
    df79 = tmp[tmp.U < 0.80]
    tmp =   df[df.U >= 0.80]
    df80 = tmp[tmp.U < 0.81]
    tmp =   df[df.U >= 0.81]
    df81 = tmp[tmp.U < 0.82]
    tmp =   df[df.U >= 0.82]
    df82 = tmp[tmp.U < 0.83]
    tmp =   df[df.U >= 0.83]
    df83 = tmp[tmp.U < 0.84]
    tmp =   df[df.U >= 0.84]
    df84 = tmp[tmp.U < 0.85]
    tmp =   df[df.U >= 0.85]
    df85 = tmp[tmp.U < 0.86]
    tmp =   df[df.U >= 0.86]
    df86 = tmp[tmp.U < 0.87]
    tmp =   df[df.U >= 0.87]
    df87 = tmp[tmp.U < 0.88]
    tmp =   df[df.U >= 0.88]
    df88 = tmp[tmp.U < 0.89]
    tmp =   df[df.U >= 0.89]
    df89 = tmp[tmp.U < 0.90]
    tmp =   df[df.U >= 0.90]
    df90 = tmp[tmp.U < 0.91]
    tmp =   df[df.U >= 0.91]
    df91 = tmp[tmp.U < 0.92]
    tmp =   df[df.U >= 0.92]
    df92 = tmp[tmp.U < 0.93]
    tmp =   df[df.U >= 0.93]
    df93 = tmp[tmp.U < 0.94]
    tmp =   df[df.U >= 0.94]
    df94 = tmp[tmp.U < 0.95]
    tmp =   df[df.U >= 0.95]
    df95 = tmp[tmp.U < 0.96]
    tmp =   df[df.U >= 0.96]
    df96 = tmp[tmp.U < 0.97]
    tmp =   df[df.U >= 0.97]
    df97 = tmp[tmp.U < 0.98]
    tmp =   df[df.U >= 0.98]
    df98 = tmp[tmp.U < 0.99]

    odfs = [
    ('70', df70),
    ('71', df71),
    ('72', df72),
    ('73', df73),
    ('74', df74),
    ('75', df75),
    ('76', df76),
    ('77', df77),
    ('78', df78),
    ('79', df79),
    ('80', df80),
    ('81', df81),
    ('82', df82),
    ('83', df83),
    ('84', df84),
    ('85', df85),
    ('86', df86),
    ('87', df87),
    ('88', df88),
    ('89', df89),
    ('90', df90),
    ('91', df91),
    ('92', df92),
    ('93', df93),
    ('94', df94),
    ('95', df95),
    ('96', df96),
    ('97', df97),
    ('98%', df98)
    ]

    results = []
    if five_p:
        fs = fdfs
    elif one_p:
        fs = odfs
    else:
        fs = dfs

    for (label, d) in fs:
        results.append(trend(label, d))

    df_all = df
    df = pd.DataFrame(results, \
            columns=['zone', 'avg_rm', 'avg_sim', 'avg_rta', 'avg_dm', 'nsim', 'nrta', 'ndm', \
                    'navg_sim', 'navg_rta', 'navg_dm', \
                    'avg_rfj_rm', 'avg_rfj_dm', 'avg_rfj_erd', 'avg_afj_rm', 'avg_afj_dm', 'avg_afj_erd'])

    df_num = df[['zone', 'nsim', 'nrta', 'ndm']]
    df_avg = df[['zone', 'avg_sim', 'avg_rta', 'avg_dm']]
    df_fast = df[['zone', 'navg_sim', 'navg_rta', 'navg_dm']]
    df_jt = df[['zone', 'avg_rm', 'avg_rfj_rm', 'avg_rfj_dm', 'avg_rfj_erd', 'avg_afj_rm', 'avg_afj_dm', 'avg_afj_erd']]

    return df_all, df, df_num, df_avg, df_fast, df_jt, fs

def do_n_plt(f, df):
    pyplot.plot(df.zone, df.nsim, 'b+-', label='ERD SIM')
    pyplot.plot(df.zone, df.nrta, 'ro-', label='ERD RTA')
    pyplot.plot(df.zone, df.ndm, 'g*-', label='DM')

    pyplot.ylim([0.55, 1.02])

    fn = os.path.splitext(os.path.basename(f))[0]
    title = 'ratio of shortened task ... ' + fn
    # pyplot.title(title)
    pyplot.legend() 
    
    if five_p:
        pyplot.savefig(fn + '_n_5.png')
    else:
        pyplot.savefig(fn + '_n.png')
    if show_plt==True:
        pyplot.show()

def do_a_plt(f, df):
    pyplot.plot(df.zone, df.avg_sim, 'b+-', label='ERD SIM')
    pyplot.plot(df.zone, df.avg_rta, 'ro-', label='ERD RTA')
    pyplot.plot(df.zone, df.avg_dm, 'g*-', label='DM')

    pyplot.ylim([0.0, 0.55])

    fn = os.path.splitext(os.path.basename(f))[0]
    title = 'shortened ratio ...' + fn
    # pyplot.title(title)
    pyplot.legend() 
    
    if five_p:
        pyplot.savefig(fn + '_f_5.png')
    else:
        pyplot.savefig(fn + '_f.png')
    if show_plt==True:
        pyplot.show()

def do_s_plt(f, df):
    pyplot.plot(df.zone, df.navg_sim, 'b+-', label='ERD SIM')
    pyplot.plot(df.zone, df.navg_rta, 'ro-', label='ERD RTA')
    pyplot.plot(df.zone, df.navg_dm, 'g*-', label='DM')

    fn = os.path.splitext(os.path.basename(f))[0]
    title = 'shortened ratio ...' + fn
    # pyplot.title(title)
    pyplot.legend() 
    
    if five_p:
        pyplot.savefig(fn + '_s_5.png')
    else:
        pyplot.savefig(fn + '_s.png')
    if show_plt==True:
        pyplot.show()

def do_jitter_plt(f, df):
    pyplot.plot(df.zone, df.avg_afj_erd, 'b+-', label='AFJ ERD')
    pyplot.plot(df.zone, df.avg_afj_dm, 'g*-', label='AFJ DM')
    pyplot.plot(df.zone, df.avg_afj_rm, 'co-', label='AFJ RM')

    pyplot.ylim([0.0, 70.00])

    fn = os.path.splitext(os.path.basename(f))[0]
    title = 'AFJ' + fn
    pyplot.legend() 
    
    if five_p:
        pyplot.savefig(fn + '_jt_5.png')
    else:
        pyplot.savefig(fn + '_jt.png')
    if show_plt==True:
        pyplot.show()

def samarize(f):
    df_all, df, df_num, df_avg, df_fast, df_jt, dfs = construct_df(f)
    do_n_plt(f, df_num)
    do_a_plt(f, df_avg)
    # do_s_plt(f, df_fast)
    do_jitter_plt(f, df_jt)
    return df_all, df, df_num, df_avg, df_fast, df_jt, dfs

def gen_test_sets():
    tss_98 = get_task_sets(TEST_N/10, 0.97)
    print("got 98")
    tss_96 = get_task_sets(TEST_N, 0.96)
    print("got 96")
    tss_94 = get_task_sets(TEST_N, 0.94)
    print("got 94")
    tss_92 = get_task_sets(TEST_N, 0.92)
    print("got 92")
    tss_90 = get_task_sets(TEST_N, 0.90)
    print("got 90")
    tss_88 = get_task_sets(TEST_N, 0.88)
    print("got 88")
    tss_86 = get_task_sets(TEST_N, 0.86)
    print("got 86")
    tss_84 = get_task_sets(TEST_N, 0.84)
    print("got 84")
    tss_82 = get_task_sets(TEST_N, 0.82)
    print("got 82")
    tss_80 = get_task_sets(TEST_N, 0.80)
    print("got 80")
    tss_78 = get_task_sets(TEST_N, 0.78)
    print("got 78")
    tss_76 = get_task_sets(TEST_N, 0.76)
    print("got 76")
    tss_74 = get_task_sets(TEST_N, 0.74)
    print("got 74")
    tss_72 = get_task_sets(TEST_N, 0.72)
    print("got 72")
    tss_70 = get_task_sets(TEST_N, 0.70)
    print("got 70")

    tss = [ tss_70, tss_72, tss_74, tss_76, tss_78, \
            tss_80, tss_82, tss_84, tss_86, tss_88, \
            tss_90, tss_92, tss_94, tss_96, tss_98 ]

    with open("experiment_tasks_010.bin", "wb") as f:
        dill.dump(tss, f)

    return tss

def gen_mini_sets():
    tss_70 = get_task_sets(12, 0.80)
    tss_75 = get_task_sets(12, 0.85)
    tss_80 = get_task_sets(12, 0.90)

    tss = [ tss_70, tss_75, tss_80 ]

    return tss

def get_num_of_task(dfs):
    ttotal = 0
    tn = 0
    for d in dfs:
        total = 0
        for t in d[1].priv:
            total += int(t.replace('t', ''))
        ttotal += total
        tn += len(d[1].priv)
        print( d[0], total / len(d[1].priv))
    print( ttotal / tn )

if __name__ == "__main__":

    # generate test set
    if False:
        if test_mini:
            tss = gen_mini_sets()
            fn = "experiment_tasks_mini.bin"
        else:
            tss = gen_test_sets()
            fn = "experiment_tasks_010.bin"

        with open(fn, "wb") as f:
            dill.dump(tss, f)

    # experiment
    if True:
        if test_mini:
            tasks_fn = "experiment_tasks_mini.bin"
            result_fn = "mini.csv"
        else:
            tasks_fn = "experiment_tasks_010.bin"
            result_fn = "result_010.csv"
        
        with open(tasks_fn, "rb") as f:
            tss = dill.load(f)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        for i, ts in enumerate(tss):
            for j in range(0, len(ts), 4):
                print("test %d - %dth" % (i, j), flush=True)
                f1 = executor.submit(test,ts[j], ts[j].ts[-1])
                f2 = executor.submit(test,ts[j+1], ts[j+1].ts[-1])
                f3 = executor.submit(test,ts[j+2], ts[j+2].ts[-1])
                f4 = executor.submit(test,ts[j+3], ts[j+3].ts[-1])
                f1.result()
                f2.result()
                f3.result()
                f4.result()
            
        # test mid priority
        if False:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            for i, ts in enumerate(tss):
                for j, t in enumerate(ts):
                    tlen = len(t.ts)
                    pn = int((tlen+1)/2) - tlen % 2
                    if tlen < 2:
                        pn = tlen
                    nt = TaskSystem(task_set = t.ts, priv_t='t'+str(pn))
                    print("test %d - %d (%d th/ %d tasks" % (i, j, pn+1, len(nt.ts)), flush=True)
                    executor.submit(test, nt, t.ts[pn])
            
    df = get_df()
    df.to_csv(result_fn)

    samarize(result_fn)
