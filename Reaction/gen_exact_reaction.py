import numpy as np
import matplotlib.pyplot as plt

def g(x):
    """Initial condition, string --> function."""

    return np.exp(-(x - np.pi) ** 2 / (2 * (np.pi / 4) ** 2))

def exact_u(x, t, ruo):
    temp1 = g(x)*np.exp(ruo*t)
    return temp1/(temp1+1-g(x))

if __name__ == '__main__':

    number_x = 256
    number_t = 100
    ruo = 0.5
    h = 2 * np.pi / number_x
    x = np.arange(0, 2 * np.pi, h)  # not inclusive of the last point
    t = np.linspace(0, 1, number_t).reshape(-1, 1)
    TT, XX = np.meshgrid(t, x)
    X = np.column_stack((np.ravel(TT), np.ravel(XX)))
    # 计算每个时空点的真实解
    u_exact = np.array([exact_u(xi, ti, ruo) for ti, xi in X]).reshape(len(x), len(t))
    # np.savez(f'Reaction_usol_ruo={float(ruo)}.npz', t=t, x=x, u=u_exact)
    plt.figure(figsize=(5, 4), dpi=150)
    plt.pcolor(TT, XX, u_exact, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.savefig(f"ruo={float(ruo)}_reference_solution.png")
    plt.close()
