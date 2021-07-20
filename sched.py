from math import ceil, floor
from prim_task import Task, Server, get_task_set, get_vss, show_task
from util import lcm_list
from collections import namedtuple
import sys

_verbose = True
_verbose = False
_stop = False
_stop = True

STOP_COUNT = 100

# MAX_PERIOD = 1000000
MAX_PERIOD = 100000
SERVER_NAME = 'VS'

# tm = namedtuple('TaskMeet', ('wcrt_sim', 'T', 'wcrt_tick'))
sr = namedtuple('SchedResult', ('TaskMeet', 'meet', 'first_deadline_tick'))
class TaskMeet(object):
    def __init__(self, wcrt_sim, period, wcrt_tick, afj, rfj):
        self.wcrt_sim = wcrt_sim
        self.period = period
        self.wcrt_tick = wcrt_tick
        self.afj = afj
        self.rfj = rfj
    
    def __repr__(self):
        return "(WCRT %d/T=%d @ tick %d, AFJ %d, RFJ %d)" % (
                self.wcrt_sim, self.period, self.wcrt_tick, self.afj, self.rfj)

class _Task(object):
    """ _Task represents task, who has wcet, period, deadline, etc, itself. 
    Class _Task is not intended to use from external file.              
    This class is only used from scheduler class Sched and manages      
    context ( what is current priority, how CPU was used, etc, of task.
    """
    def __init__(self, name, wcet, period, deadline):
        self.name = name
        self.wcet = wcet
        self.period = period
        self.deadline = deadline
        self.release_time = 0
        self.wcrt = self.wcet
        self.wcrt_tick = 0
        self.bcrt = self.deadline   # best case response time
        self.consumed = 0
        self.base_priority = 0
        self.priority = 0
        self.pjft = 0           # previours finish time of job
        self.afj = 0            # ablolute finishing jitter
        self.rfj = 0            # relative finishing jitter

    def utilization(self):
        return self.wcet / self.period

    def absolute_deadline(self, tick):
        return int(ceil( (tick + 1.0) / self.period - 1) * self.period) + self.deadline

    def __repr__(self):
        return "_Task(%s, wcet=%d, period=%d, consumed=%d, priority=%d, wcrt=%d, bcrt=%d)" % (
                self.name, self.wcet, self.period,
                self.consumed, self.priority, self.wcrt, self.bcrt)


def debug_print(str):
    if _verbose:
        print( str )

def _show_q(q):
    if _verbose:
        print( "\tq -> " )
        for t in q:
            print( "%s(%d)[%d/%d]" % (t.name, t.priority, t.consumed, t.wcet))
        print( "" )


class Sched(object):
    """ Sched class is a scheduler ones scheduling policy is Rate Monotonic """
    def __init__(self, taskset):
        self.rdy_q = []
        self.slp_q = []
        self.tasks = []  # set of _Task
        self.tick = 0
        self.meet = True

        self.vs = None  # not used, needed for deadline check
        self.first_deadline_t = 0

        T = []
        for t in taskset:
            T.append(t.T)
        self.hyper_period = lcm_list(T)
        if self.hyper_period > MAX_PERIOD:
            self.hyper_period = MAX_PERIOD


        # init _Task set and ready que
        for t in taskset:
            _t = _Task(t.name, t.C, t.T, t.D)
            self.tasks.append(_t)
            self.rdy_q.append(_t)

    def _is_idle(self):
        """ Check whether rdy_q is empty """
        if len(self.rdy_q) == 0:
            debug_print( "period : %d : idle" % self.tick )
            return True
        return False

    def _check_activate(self):
        """ Wakeup task whose period equal to tick """
        activated = False
        ts = []
        for t in self.slp_q:
            if (self.tick % t.period) == 0:
                t.release_time = self.tick
                t.consumed = 0
                ts.append(t)
                activated = True

        for t in ts:
            debug_print( "\tactivate %s" % t.name )
            t.priority = t.base_priority
            self.slp_q.remove(t)
            self.rdy_q.append(t)

        return activated

    def _consume(self, run_t, force=False):
        """ Consume capacity, Change task status to sleep if it runs equal to wcet
            Returns True if the task finishes it's execution """
        run_t.consumed += 1
        debug_print( "\t\t%s runs (remain %d / %d)" \
                     % (run_t.name, run_t.consumed, run_t.wcet) )
        if force or (run_t.consumed == run_t.wcet) :
            debug_print( "\t\t%s -> slp" % run_t.name )
            rt = self.tick - run_t.release_time + 1
            debug_print( "\t\trt -> %d" % rt )
            # update best/worst case response time
            if run_t.wcrt < rt:
                run_t.wcrt = rt
                run_t.wcrt_tick = self.tick
            if run_t.bcrt > rt:
                run_t.bcrt = rt

            # update jitter value
            run_t.afj = run_t.wcrt - run_t.bcrt
            if run_t.pjft == 0:
                run_t.pjft = rt
            else:
                rj = abs(rt - run_t.pjft)
                if run_t.rfj < rj:
                    run_t.rfj = rj

            self.rdy_q.pop(self.rdy_q.index(run_t))    # remove run_t
            self.slp_q.append(run_t)

            return True
        else:
            return False

    def _check_deadline_misses(self):
        """ Check deadline misses for all tasks """
        for t in self.rdy_q:
            d = t.absolute_deadline(self.tick) - 1
            if (d == self.tick) and (t.consumed < t.wcet):
                if t == self.vs:
                    self.rdy_q.pop(self.rdy_q.index(t))    # remove run_t
                    self.slp_q.append(t)
                    continue
                else:
                    debug_print( "\t\tDEADLINE MISS!! %s" % t.name )
                    self.meet = False
                    self.rdy_q.pop(self.rdy_q.index(t))    # remove run_t
                    self.slp_q.append(t)
                return t
        return None

    def _schedule(self):
        """ Set priority by RM rule ( shorter period has higer priority ) """
        self.rdy_q = sorted(self.rdy_q, key=lambda t:t.deadline)

    def generate_sim(self):
        """ Simulator """
        self._schedule()

        stop_cnt = STOP_COUNT

        for i in range(self.hyper_period):
            if _stop and not self.meet:
                stop_cnt -= 1
                if stop_cnt == 0:
                    yield None

            self.tick = i
            debug_print( "\n=== period %d ===" % i )

            if self._check_activate():
                self._schedule()

            if self._is_idle():
                yield (i, 'idle', '')
            else:
                _show_q(self.rdy_q)
                run_t = self.rdy_q[0]
                self._consume(run_t)
                debug_print( "period : %d : %s" % (i, run_t.name) )
                dt = self._check_deadline_misses()
                if dt:
                    if self.first_deadline_t == 0:
                        self.first_deadline_t = i
                    yield (i, run_t.name, 'DM %s' % dt.name)
                else:
                    yield (i, run_t.name, '')

        # all sim finished
        yield None

    def do_sim(self):
        """ Simulate until hyper period """
        g = self.generate_sim()
        while True:
            if g.__next__() == None:
                break;
    
    def _get_wcrt(self, duration=MAX_PERIOD):
        """ Retrieve worst case response time by actual simulation.
        Simulation is performed until tick is equal to specified duration or
        is reached to hyper period of taskset.
        """
        g = self.generate_sim()

        if duration == 0:
            while True:
                if g.__next__() == None:
                    break;
        else:
            for i in range(0, duration):
                if g.__next__() == None:
                    break;

        ts = []
        for i in range(len(self.tasks)):
            t = self.tasks[i]
            # ts.append(tm(t.wcrt, t.period, t.wcrt_tick))
            # ts.append((t.wcrt, t.period, t.wcrt_tick))
            ts.append(TaskMeet(t.wcrt, t.period, t.wcrt_tick, t.afj, t.rfj))
        return sr(ts, self.meet, self.first_deadline_t)

    
    def _get_sched_list(self, offset, period):
        g = self.generate_sim()

        l = [g.__next__() for x in range(offset+period)]
        return l[offset:offset+period]

    @staticmethod
    def get_sched_list(taskset, offset, period):
        """ Retrieve sched chart """
        s = Sched(taskset)
        return s._get_sched_list(offset, period)
    
    @staticmethod
    def get_wcrt(taskset, duration=MAX_PERIOD):
        """ Retrieve sched chart """
        s = Sched(taskset)
        return s._get_wcrt(duration)

class SchedErd(Sched):
    """ SchedErd class is a scheduler ones scheduling policy is ERD,
    Execution Right Delegation. """
    def __init__(self, taskset, vs):
        super(SchedErd, self).__init__(taskset)

        # init priorities and tasksets
        self.tasks = sorted(self.tasks, key=lambda t:t.deadline)
        i = 0
        self.priv_t = None

        for t in self.tasks:
            t.base_priority = t.priority = i + 1
            i += 2
            if vs.priv_task == t.name:
                self.priv_t = t
        
        self.tasks = sorted(self.tasks, key=lambda t:t.period)

        self.vs = _Task(SERVER_NAME, vs.C, vs.T, vs.T)
        self.vs.base_priority = self.vs.priority = self.__get_pri(self.tasks,
                self.vs)
        self.tasks.append(self.vs)
        self.rdy_q.append(self.vs)

        if vs.C == vs.T:
            self.optimal = True
        else:
            self.optimal = False

    def __get_pri(self, q, vs):
        for t in q:
            if t.period < vs.period:
                continue
            return t.priority - 1

    def _is_idle(self):
        """ Check whether rdy_q is empty.
        For ERD case, the case that VS is a only active task is handled as
        idle but consumes VS's capacity
        """
        if len(self.rdy_q) == 0:
            debug_print( "period : %d : idle" % self.tick )
            return True
        elif (self.rdy_q[0] == self.vs) and (len(self.rdy_q) == 1):
            self._consume(self.vs, force=True)
            return True

        return False

    def _schedule(self):
        """ Set priority by priority itself and update rdy_q.
        Compared with RM, ERD uses specified priority
        """
        self.rdy_q = sorted(self.rdy_q, key=lambda t:t.priority)

    def generate_sim(self):
        """ Simulator """

        # set priority by priority
        self._schedule()

        stop_cnt = STOP_COUNT

        for i in range(self.hyper_period):
            if _stop and not self.meet:
                stop_cnt -= 1
                if stop_cnt == 0:
                    yield None

            self.tick = i
            debug_print( "\n=== period %d ===" % i )

            if self._check_activate():
                self._schedule()

            if self._is_idle():
                debug_print( "period : %d : idle" % i)
                yield (i, 'idle', '')
                continue

            _show_q(self.rdy_q)
            _show_q(self.slp_q)
            run_t = self.rdy_q[0]

            delg_t = ''
            if run_t == self.vs:
                try:
                    # is priv task activated?
                    self.rdy_q.index(self.priv_t)

                    if self.vs.priority == self.vs.base_priority:
                        delg_t = self.vs.name
                    else:
                        delg_t = self.rdy_q[1].name
                    self._consume(self.vs)
                    run_t = self.priv_t
                except ValueError:
                    # as priority exchange rule, server priority down to
                    # ready task
                    run_t = self.rdy_q[1]
                    if not self.optimal:
                        self.vs.priority = run_t.priority - 1
                        self._schedule()

            self._consume(run_t)
            dt = self._check_deadline_misses()

            debug_print("period : %d : %s" % (i, run_t.name))

            if dt:
                if self.first_deadline_t == 0:
                    self.first_deadline_t = i
                yield (i, run_t.name, 'DM %s' % dt.name)
            else:
                yield( i, run_t.name, delg_t )

        # all sim finished
        yield None

    @staticmethod
    def get_sched_list(taskset, vs, offset, period):
        """ Retrieve sched chart """
        s = SchedErd(taskset, vs)
        return s._get_sched_list(offset, period)
    
    @staticmethod
    def get_wcrt(taskset, vs, duration=MAX_PERIOD):
        s = SchedErd(taskset, vs)
        return s._get_wcrt(duration)

class SchedErd2(Sched):
    """ SchedErd2 class is a scheduler ones scheduling policy is ERD 2.0,
    Execution Right Delegation. """
    def __init__(self, taskset, vss):
        super(SchedErd2, self).__init__(taskset)
        self.vss = vss
        self.itf = [0] * len(self.tasks)
        self.max_itf = [0] * len(self.tasks)

        for i, t in enumerate(self.tasks):
            t.base_priority = t.priority = i

        self.priv_t = None
        for t in self.tasks:
            if vss[0].priv_task == t.name:
                self.priv_t = t

        for vs in vss:
            for i, t in enumerate(self.tasks):
                if t.period == vs.T:
                    self.max_itf[i] = vs.C
    
    def _consume(self, run_t):
        if super(SchedErd2, self)._consume(run_t) == True:
            self.itf[run_t.priority] = 0

    def is_delegate(self, run_t):
        try:
            # is priv task activated?
            self.rdy_q.index(self.priv_t)

            for t in self.rdy_q:
                if (t != self.priv_t) and (self.itf[t.priority] >= self.max_itf[t.priority]):
                    return False

            return True

        except ValueError:
            # do notiong
            debug_print( "\t\tpriv_t is not in rdy_q" )
            return False
    
    def generate_sim(self):
        """ Simulator """
        self._schedule()
        
        stop_cnt = STOP_COUNT

        for i in range(self.hyper_period):
            if _stop and not self.meet:
                stop_cnt -= 1
                if stop_cnt == 0:
                    yield None

            self.tick = i
            debug_print( "\n=== period %d ===" % i )

            if self._check_activate():
                self._schedule()

            if self._is_idle():
                yield (i, 'idle', '')
            else:
                _show_q(self.rdy_q)
                delg_t = ''
                run_t = self.rdy_q[0]
                if (run_t != self.priv_t) and (self.is_delegate(run_t)):
                    # priv_t could run instead of run_t
                    delg_t = run_t.name
                    run_t = self.priv_t
                    for t in self.rdy_q :
                        self.itf[t.priority] += 1

                self._consume(run_t)
                debug_print( "period : %d : %s" % (i, run_t.name) )
                dt = self._check_deadline_misses()
                if dt:
                    if self.first_deadline_t == 0:
                        self.first_deadline_t = i
                    yield (i, run_t.name, 'DM %s' % dt.name)
                else:
                    yield( i, run_t.name, delg_t )

        # all sim finished
        yield None
    
    @staticmethod
    def get_sched_list(taskset, vss, offset, period):
        """ Retrieve sched chart """
        s = SchedErd2(taskset, vss)
        return s._get_sched_list(offset, period)
    
    @staticmethod
    def get_wcrt(taskset, vss, duration=MAX_PERIOD):
        s = SchedErd2(taskset, vss)
        return s._get_wcrt(duration)

def plot(ts, vs, offset, period):
    """ Draw scheduling chart """
    vts = []
    if vs == None:
        sched_chart = Sched.get_sched_list(ts, offset, period)
        for t in ts:
            vts.append(t)
    elif type(vs) is list:
        sched_chart = SchedErd2.get_sched_list(ts, vs, offset, period)
        for t in ts:
            vts.append(t)
    else:
        sched_chart = SchedErd.get_sched_list(ts, vs, offset, period)

        has_vs = False
        for t in ts:
            if (has_vs == False) and (vs.T <= t.T):
                vts.append(Task(SERVER_NAME, vs.C, vs.T, vs.T))
                vts.append(t)
                has_vs = True
            else:
                vts.append(t)

    simlen = len(sched_chart)

    sys.stdout.write("  \t: ")
    for i in range(0, int(simlen/10 + 1)):
        sys.stdout.write("%05d               " % (offset + i*10))
    sys.stdout.write('\n')
    sys.stdout.write("  \t: ")
    for i in range(0, int(simlen/10 + 1)):
        sys.stdout.write("|                   ")
    sys.stdout.write('\n')

    miss = 0
    miss_task = None

    for t in vts:
        sys.stdout.write("%s\t: " % t.name)
        for i in range(0, simlen):
            if ((offset+i) % t.T) == 0:
                sys.stdout.write("|")
            else:
                sys.stdout.write(" ")
            v = sched_chart[i]
            if ('DM' in v[2]) and (t.name in v[2]):
                sys.stdout.write('!')
                if miss == 0:
                    miss = i
                    miss_task = t.name
            elif v[1] == t.name:
                sys.stdout.write('X')
            elif v[2] == t.name:
                sys.stdout.write('v')
            else:
                sys.stdout.write('_')
        sys.stdout.write('\n')

    if miss:
        sys.stdout.write("  \t  ")
        for j in range(0, miss):
            sys.stdout.write("  ")
        sys.stdout.write(" A\n")

        sys.stdout.write("  \t  ")
        for j in range(0, miss):
            sys.stdout.write("  ")
        sys.stdout.write(' |\n')
        sys.stdout.write("  \t  ")
        for j in range(0, miss):
            sys.stdout.write("  ")
        sys.stdout.write(" -------- DEADLINE MISS task %s\n" % miss_task)

    sys.stdout.write("\n")


if __name__ == "__main__":
    # ts = get_task_set('./sampleTasks/erd_success.txt')
    # ts = get_task_set('./sampleTasks/pt5.txt')
    # ts = get_task_set('./sampleTasks/cata.txt')
    # ts = get_task_set('../beyond/task_set_erd_miss_20/t2591')
    # vss = get_vss(ts, 't4')
    # sched = SchedErd(ts, vss[0])
    # sched.do_sim()

    # sched = SchedErd2(ts, vss)
    # sched.do_sim()
    # sl = SchedErd2.get_sched_list(ts, vss, 0, 32)
    
    # print( Sched.get_wcrt(ts, 0))
    # print( SchedErd.get_wcrt(ts, vss[0], 0))
    # print( SchedErd2.get_wcrt(ts, vss, 0))
     
    # ts = get_task_set('./test_tasks/t1')
    # vss = get_vss(ts, 't3')
    # rm_sched = Sched(ts)
    # erd_sched = SchedErd(ts, vss[0])
    # res = SchedErd.get_wcrt(ts, vss[0], 8800)
    dm = get_task_set('./sampleTasks/dm.txt')
