import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parametrar
R_val = 1.0
L_val = 2.0
C_val = 0.5
t_span = (0, 20)
y0 = [1.0, 0.0]  # q0=1, i0=0

"""
T1.a
"""

def rlc_system(t, y, R, L, C):
    q = y[0] 
    i = y[1] 
    dq_dt = i
    di_dt = -(R * i + (1/C) * q) / L
    return [dq_dt, di_dt]

def euler_framåt(f, t_span, y0, h, *args):
    t_start, t_end = t_span
    # Beräkna antag tidssteg
    N = int(round((t_end - t_start) / h))
    t = np.linspace(t_start, t_end, N + 1)
    # Skapa en matris för värdena av q, med två rader för q, i och t kolumner
    num_vars = len(y0)
    y = np.zeros((num_vars, N + 1))
    y[:, 0] = y0

    for k in range(N):
        dydt = np.array(f(t[k], y[:, k], *args))
        y[:, k+1] = y[:, k] + h * dydt
    return t, y

"""
T1.b
"""

def ivp_lösare(rlc_system, tidspann, initialvärde, R, L, C):
    solution = solve_ivp(
        fun=rlc_system, 
        t_span=tidspann,
        y0=initialvärde,
        method="RK45",
        args=(R, L, C),
    )
    return solution

"""
T1.c
"""

# dämpad svängning
solution_1 = ivp_lösare(rlc_system,t_span,y0,R=1,L=1,C=0.5)
# odämpad svängning
solution_2 = ivp_lösare(rlc_system,t_span,y0,R=0,L=1,C=0.5)

#plott av ivp:n
plt.figure(figsize = (12, 8))
plt.plot(solution_1.t, solution_1.y[0], label="dämpad svängning")
plt.plot(solution_2.t, solution_2.y[0], label="odämpad svängning")
plt.xlabel('t')
plt.ylabel('q')
plt.legend(loc = 2)
plt.grid(True)
plt.show() 

"""
T1.d
"""

N_values = [20, 40, 80, 160]
plt.figure(figsize=(12, 8))

for N in N_values:
    h = (t_span[1] - t_span[0]) / N
    t_euler, y_euler = euler_framåt(rlc_system, t_span, y0, h, R_val, L_val, C_val)
    plt.plot(t_euler, y_euler[0, :], label=f'Euler N={N}')

plt.plot(solution_1.t, solution_1.y[0], 'k--', linewidth=2, label='Referens (RK45)')
plt.xlabel("t")
plt.ylabel("q")
plt.ylim(-2, 2) # Behövs för se resultatet bättre
plt.legend(loc = 2)
plt.grid(True)
plt.show()

"""
T1.e
"""

referens = solve_ivp(rlc_system, t_span, y0, args=(R_val, L_val, C_val), rtol=1e-12, atol=1e-12)
y_exakt_T = referens.y[:, -1] 

# Vi börjar vid N=40 för det där då Euler blir stabil
N_values = [40, 80, 160, 320, 640, 1280]
errors_q = []
errors_i = []

print(f"{'N':<5} {'h':<10} {'Error q(T)':<15} {'Error i(T)':<15}")
print("-" * 50)

# fel för olika N
for N in N_values:
    h = (t_span[1] - t_span[0]) / N
    
    t_euler, y_euler = euler_framåt(rlc_system, t_span, y0, h, R_val, L_val, C_val)
    
    y_approx_T = y_euler[:, -1]
    
    # fel = |y_approx - y_exakt|
    e_q = np.abs(y_approx_T[0] - y_exakt_T[0])
    e_i = np.abs(y_approx_T[1] - y_exakt_T[1])

    errors_q.append(e_q)
    errors_i.append(e_i)
    
    print(f"{N:<5} {h:<10.4f} {e_q:<15.6e} {e_i:<15.6e}")

# Bestäm order p 
# formel: p approx log2( error(h) / error(h/2) )
print(f"{'N_pair':<15} {'p for q':<15} {'p for i':<15}")

for k in range(len(N_values) - 1):
    ratio_q = errors_q[k] / errors_q[k+1]
    ratio_i = errors_i[k] / errors_i[k+1]
    
    p_q = np.log2(ratio_q)
    p_i = np.log2(ratio_i)
    
    pair_label = f"{N_values[k]}/{N_values[k+1]}"
    print(f"{pair_label:<15} {p_q:<15.4f} {p_i:<15.4f}")

