# erd_single
ERD Scheduling Algorithm Simulator for Single Processor

## Simple task set experiment
For simple task set experiment, do

    $ipython
    In [1]: run ./tasks.py

You can see scheduling plot of ERD like this:

    In [2]: pt5.plot()
            : 00000               00010               00020               00030
            : |                   |                   |                   |
    VS      : |v _ _ _ _|v _ _ _ _|v _ _ _ _|v _ _ _ _|v _ _ _ _|_ _ _ _ _|v _
    t1      : |_ X _ _ _|_ X _ _ _|_ X _ _ _|_ X _ _ _|_ X _ _ _|X _ _ _ _|_ X
    t2      : |_ _ X _ _ _|_ X _ _ _ _|X _ _ _ _ _|X _ _ _ _ _|X _ _ _ _ _|_ _
    t3      : |_ _ _ X X _ _ _ _|X _ _ _ X _ _ _ _|_ X _ _ X _ _ _ _|X X _ _ _
    t4      : |X _ _ _ _ X _ _ X _ X _ _ _|X X _ X _ _ X _ _ _ _ _ _ _|_ X X _


There are several candidates of VS. When changing VS, try:

    In [6]: pt5.vss
    Out[6]:
    [VirtualServer(C=1, T=5, priv_task='t4'),
     VirtualServer(C=1, T=6, priv_task='t4'),
     VirtualServer(C=3, T=9, priv_task='t4')]

    In [7]: pt5.vs
    Out[7]: VirtualServer(C=1, T=5, priv_task='t4')

    In [8]: pt5.vs = pt5.vss[2]

    In [9]: pt5.plot()
            : 00000               00010               00020               00030
            : |                   |                   |                   |
    t1      : |X _ _ _ _|X _ _ _ _|X _ _ _ _|X _ _ _ _|X _ _ _ _|X _ _ _ _|X _
    t2      : |_ X _ _ _ _|X _ _ _ _ _|X _ _ _ _ _|X _ _ _ _ _|X _ _ _ _ _|_ X
    VS      : |_ _ v v v _ _ _ _|v _ _ _ _ v _ v _|_ v _ _ _ _ _ _ _|_ v v _ _
    t3      : |_ _ _ _ _ _ _ X X|_ _ X _ X _ _ _ _|_ _ _ X X _ _ _ _|X _ _ _ _
    t4      : |_ _ X X X _ _ _ _ X _ _ _ _|X _ X X _ X _ _ _ _ _ _ _ _|X X _ _

## Simulation with large task sets
For mini-task set experiment, do

    In [10]: run ./exam.py

For large-task set experiment, disable test_mini flag in exam.py and execute the script.

When changing task set generating parameter, modify gen.py line@191
