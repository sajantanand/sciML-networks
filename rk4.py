import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

def dydt_sho_2D(t, y):
    x, y, px, py = y
    return np.array([px, py, -1*x, -1*y])

def dydt_sho_1D(t, y):
    x, p = y
    return np.array([p, -1*x])

def dydt_anharmonic(t, y):
    x, p = y
    #return np.array([p, -1*x - x**5])
    return np.array([p, - x**5])

def rk4(dydt, y0, tspan, n):
    if (np.ndim(y0) == 0):
        m = 1
    else:
        m = len(y0)

    tfirst, tlast = tspan
    dt = (tfirst - tlast) / n
    t = np.zeros(n+1)
    y = np.zeros((n+1, m))
    t[0] = tfirst
    y[0,:] = y0

    for i in range(n):
        f1 = dydt(t[i],             y[i,:])
        f2 = dydt(t[i] + dt / 2.0,  y[i,:] + dt * f1 / 2.0)
        f3 = dydt(t[i] + dt / 2.0,  y[i,:] + dt * f2 / 2.0)
        f4 = dydt(t[i] + dt,        y[i,:] + dt * f3)

        t[i+1] = t[i] + dt
        y[i+1, :] = y[i,:] + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0

    return t, y

if __name__ == "__main__":
    t, y = rk4(dydt_sho_1D, (1,0), (0,10), 100000)
    print(y[:10, :])
    fig = plt.figure(figsize=(4,4))
    plt.plot(y[:,0], y[:,1])
    plt.xlabel('x')
    plt.ylabel('p')
    plt.savefig('sho_phase_space.pdf')
    t, y = rk4(dydt_anharmonic, (1,0), (0,10), 100000)
    print(y[:10, :])
    fig = plt.figure(figsize=(4,4))
    plt.plot(y[:,0], y[:,1])
    plt.xlabel('x')
    plt.ylabel('p')
    plt.savefig('anharmonic_phase_space.pdf')
