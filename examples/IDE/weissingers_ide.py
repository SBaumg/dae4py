import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau

def F(t, y, y_p):
    return t * y**2 * y_p**3 - y**3 * y_p**2 + t * (t**2+1) * y_p - t**2 * y

def true_sol_y(t):
    return np.sqrt(t**2+0.5)

def true_sol_y_p(t):
    return t/true_sol_y(t)

if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1
    t_span = (t0, t1)

    # initial conditions
    y0 = true_sol_y(t0)
    y_p0 = true_sol_y_p(t0)

    # number of stages
    s = 2
    
    # DAE solver
    sol=solve_dae_IRK(F, y0, y_p0, t_span=t_span, h=1e-2, tableau=radau_tableau(s), atol=1e-6, rtol=1e-6)
    t = sol.t
    y = sol.y
    y_p = sol.yp

    # true solution
    t_true = np.linspace(t0, t1, 100)
    y_true = true_sol_y(t_true)
    y_p_true = true_sol_y_p(t_true)

    # plot results
    fig, ax = plt.subplots(2,1)
    ax[0].plot(t_true, y_true, label='True solution y(t)')
    ax[0].plot(t, y, 'x', label='DAE solution y(t)')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('y(t)')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t_true, y_p_true, label='True solution y\'(t)')
    ax[1].plot(t, y_p, 'x', label='DAE solution y\'(t)')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('y\'(t)')
    ax[1].legend()
    ax[1].grid()

    plt.show()