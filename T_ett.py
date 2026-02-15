# Labb 2, T1
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp

def rlc_system(t, y, R, L, C):
    q = y[0] 
    i = y[1] 
    dq_dt = i
    di_dt = -(R * i + (1/C) * q) / L
    return [dq_dt, di_dt]


def rlc_lutning(t, y, R, L, C):
    lista = rlc_system(t, y, R, L, C)
    return lista[1]




q0 = 1.0
i0 = 0.0
y0 = [q0, i0]

t_span = (0, 20)

def ivp_lösare(rlc_system, tidspann, initialvärde, R, L, C):
    solution = solve_ivp(
        fun=rlc_system, 
        t_span=tidspann,
        y0=initialvärde,
        method="RK45",
        args=(R, L, C),
    )
    return solution



print(rlc_system(0,y0,R=1,L=1,C=0.5))


# dämpad svängning
solution_1 = ivp_lösare(rlc_system,t_span,y0,R=1,L=1,C=0.5)
# odämpad svängning
solution_2 = ivp_lösare(rlc_system,t_span,y0,R=0,L=1,C=0.5)

#plott av ivp:n
plt.figure(figsize = (12, 8))
plt.plot(solution_1.t, solution_1.y[0], "b", label="dämpad svängning")
plt.plot(solution_2.t, solution_2.y[0],"r", label="odämpad svängning")
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc = 2)
plt.grid(True)
plt.show() 
