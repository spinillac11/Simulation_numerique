import numpy as np
import matplotlib.pyplot as plt
from simul import Simul


def pressure_vs_N_and_b(sigmas, N_values, L, simul_time, K_scale, replicas):
    """
    Devuelve:
    resultados_P[sigma] = [P(N1), P(N2)...]
    resultados_b[sigma] = [b(N1), b(N2)...]
    """
    resultados_P = {sigma: [] for sigma in sigmas}
    resultados_b = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for N in N_values:

            pres = []
            ecs = []

            for _ in range(replicas):
                try:
                    sim = Simul(simul_time=simul_time, sigma=sigma, L=L, N=N, K_scale=K_scale)
                except ValueError:
                    pres.append(np.nan)
                    ecs.append(np.nan)
                    continue

                P, Ec = sim.md_step()
                pres.append(P)
                ecs.append(Ec)

            P_avg = np.nanmean(pres)
            Ec_avg = np.nanmean(ecs)

            resultados_P[sigma].append(P_avg)

            # Calcular b solo si P es válido
            if np.isnan(P_avg) or P_avg == 0:
                resultados_b[sigma].append(np.nan)
            else:
                V = L * L
                b = (V / N) - (Ec_avg / (N*P_avg))
                resultados_b[sigma].append(b)

    return resultados_P, resultados_b

def pressure_vs_L_and_b(sigmas, L_values, N, simul_time, K_scale, replicas):

    resultados_P = {sigma: [] for sigma in sigmas}
    resultados_b = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for L in L_values:

            pres = []
            ecs = []

            for _ in range(replicas):
                try:
                    sim = Simul(simul_time=simul_time, sigma=sigma, L=L, N=N, K_scale=K_scale)
                except ValueError:
                    pres.append(np.nan)
                    ecs.append(np.nan)
                    continue

                P, Ec = sim.md_step()
                pres.append(P)
                ecs.append(Ec)

            P_avg = np.nanmean(pres)
            Ec_avg = np.nanmean(ecs)

            resultados_P[sigma].append(P_avg)

            if np.isnan(P_avg) or P_avg == 0:
                resultados_b[sigma].append(np.nan)
            else:
                V = L * L
                b = (V / N) - (Ec_avg / (N*P_avg))
                resultados_b[sigma].append(b)

    return resultados_P, resultados_b

def plot_b_vs_N(sigmas, N_values, resultados_b):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        bvals = np.array(resultados_b[sigma])
        valid = ~np.isnan(bvals)
        plt.plot(np.array(N_values)[valid], bvals[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("N")
    plt.ylabel("b (excluded volume parameter)")
    plt.title("b vs N for different σ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show() 

def plot_b_vs_L(sigmas, L_values, resultados_b):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        bvals = np.array(resultados_b[sigma])
        valid = ~np.isnan(bvals)
        plt.plot(np.array(L_values)[valid]**2, bvals[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("L")
    plt.ylabel("b (excluded volume parameter)")
    plt.title("b vs L for different σ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pressure_vs_L(sigmas, L_values, resultados):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        data = np.array(resultados[sigma])
        valid = ~np.isnan(data)

        plt.plot(1/(np.array(L_values)[valid]**2), data[valid], "-o", label=f"sigma = {sigma}")

    plt.xlabel("Box size L")
    plt.ylabel("Pressure P")
    plt.title("Pressure vs L for different σ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pressure_vs_N(sigmas, N_values, resultados):

    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        data = np.array(resultados[sigma])
        valid = ~np.isnan(data)

        plt.plot(np.array(N_values)[valid], data[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("Number of particles N")
    plt.ylabel("Pressure P")
    plt.title("Pressure vs N for different σ")
    plt.grid()
    plt.legend()
    plt.show()

def main():

    simul_time = 5.0
    K_scale = 1.0
    replicas = 2

    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N_values = list(range(10, 160, 10))
    L_values = [4, 5, 6, 8, 10, 12, 15]

    # # --- P vs N + b vs N ---
    # P_N, b_N = pressure_vs_N_and_b(sigmas, N_values, L=10, simul_time=simul_time,
    #                                K_scale=K_scale, replicas=replicas)

    # plot_b_vs_N(sigmas, N_values, b_N)
    # plot_pressure_vs_N(sigmas, N_values, P_N)

    # --- P vs L + b vs L ---
    P_L, b_L = pressure_vs_L_and_b(sigmas, L_values, N=40, simul_time=simul_time,
                                   K_scale=K_scale, replicas=replicas)

    plot_b_vs_L(sigmas, L_values, b_L)
    plot_pressure_vs_L(sigmas, L_values, P_L)
    

if __name__ == "__main__":
    main()

