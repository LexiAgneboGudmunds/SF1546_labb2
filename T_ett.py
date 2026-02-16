import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parameters for T1.d [cite: 79] ---
R_val = 1.0
L_val = 2.0
C_val = 0.5
t_span = (0, 20)
y0 = [1.0, 0.0]  # q0=1, i0=0



# Your existing system function
def rlc_system(t, y, R, L, C):
    q = y[0] 
    i = y[1] 
    dq_dt = i
    di_dt = -(R * i + (1/C) * q) / L
    return [dq_dt, di_dt]

def euler_forward(f, t_span, y0, h, *args):
    """
    Euler forward method for systems of ODEs.
    
    Parameters:
    f      : Function handle for the derivative f(t, y, *args)
    t_span : Tuple (start_time, end_time)
    y0     : List or Array of initial conditions [y1_0, y2_0, ...]
    h      : Step size
    *args  : Extra arguments to pass to f (e.g., R, L, C)
    
    Returns:
    t      : Array of time points
    y      : Array of solution points (shape: num_vars x num_steps)
    """
    t_start, t_end = t_span
    
    # Calculate number of steps N
    N = int(round((t_end - t_start) / h))
    
    # Initialize time array
    t = np.linspace(t_start, t_end, N + 1)
    
    # Initialize solution matrix
    # Rows = number of variables (2 for RLC), Columns = number of time steps
    num_vars = len(y0)
    y = np.zeros((num_vars, N + 1))
    
    # Set initial conditions in the first column
    y[:, 0] = y0
    
    # Euler Loop
    for k in range(N):
        # Get derivatives at current step (t[k], y[:, k])
        # We wrap the result in np.array() to ensure vector math works
        dydt = np.array(f(t[k], y[:, k], *args))
        
        # Euler step: y_{k+1} = y_k + h * f(t_k, y_k)
        y[:, k+1] = y[:, k] + h * dydt
        
    return t, y


def ivp_lösare(rlc_system, tidspann, initialvärde, R, L, C):
    solution = solve_ivp(
        fun=rlc_system, 
        t_span=tidspann,
        y0=initialvärde,
        method="RK45",
        args=(R, L, C),
    )
    return solution



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


# List of N values to test 
N_values = [20, 40, 80, 160]

plt.figure(figsize=(12, 8))

# Loop through each N value
for N in N_values:
    # Calculate step size h based on N
    h = (t_span[1] - t_span[0]) / N
    
    # Call our new vector-capable Euler function
    t_euler, y_euler = euler_forward(rlc_system, t_span, y0, h, R_val, L_val, C_val)
    
    # Plot Charge q(t) (which is the first row, y_euler[0, :])
    plt.plot(t_euler, y_euler[0, :], label=f'Euler N={N} (h={h:.3f})')

# --- Add Reference Solution (solve_ivp) ---
# We use solve_ivp as the "exact" reference as requested in [cite: 82]
sol_ref = solve_ivp(rlc_system, t_span, y0, method="RK45", args=(R_val, L_val, C_val), rtol=1e-9)
plt.plot(sol_ref.t, sol_ref.y[0], 'k--', linewidth=2, label='Reference (solve_ivp)')

# Formatting
plt.title("RLC Circuit: Euler's Method Stability Analysis (T1.d)")
plt.xlabel("Time t")
plt.ylabel("Charge q(t)")
plt.ylim(-2, 2) # Limit y-axis to see stability better
plt.legend()
plt.grid(True)
plt.show()

# Get the "Exact" solution using solve_ivp with high precision
ref_sol = solve_ivp(rlc_system, t_span, y0, args=(R_val, L_val, C_val), rtol=1e-12, atol=1e-12)
y_exact_T = ref_sol.y[:, -1]  # The values [q(20), i(20)]

# --- 2. Convergence Loop ---
# Start at N=40 (stable region) and double it 4 times
N_values = [40, 80, 160, 320, 640, 1280]
errors_q = []
errors_i = []

print(f"{'N':<5} {'h':<10} {'Error q(T)':<15} {'Error i(T)':<15}")
print("-" * 50)

for N in N_values:
    h = (t_span[1] - t_span[0]) / N
    
    # Run Euler's Method
    t_euler, y_euler = euler_forward(rlc_system, t_span, y0, h, R_val, L_val, C_val)
    
    # Extract the solution at the final time T (last column)
    y_approx_T = y_euler[:, -1]
    
    # Calculate absolute error for each component separately
    # Error = |y_approx - y_exact|
    e_q = np.abs(y_approx_T[0] - y_exact_T[0])
    e_i = np.abs(y_approx_T[1] - y_exact_T[1])
    
    errors_q.append(e_q)
    errors_i.append(e_i)
    
    print(f"{N:<5} {h:<10.4f} {e_q:<15.6e} {e_i:<15.6e}")

# --- 3. Calculate Order of Accuracy (p) ---
# Formula: p approx log2( error(h) / error(h/2) )
print("\n--- Empirical Order of Accuracy (p) ---")
print(f"{'N_pair':<15} {'p for q':<15} {'p for i':<15}")

for k in range(len(N_values) - 1):
    # Ratio of errors between step h and h/2
    ratio_q = errors_q[k] / errors_q[k+1]
    ratio_i = errors_i[k] / errors_i[k+1]
    
    # Calculate p
    p_q = np.log2(ratio_q)
    p_i = np.log2(ratio_i)
    
    pair_label = f"{N_values[k]}/{N_values[k+1]}"
    print(f"{pair_label:<15} {p_q:<15.4f} {p_i:<15.4f}")

