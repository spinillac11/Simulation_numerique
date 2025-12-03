import numpy as np
import matplotlib.pyplot as plt
from simul import Simul

# Experimental functions to compute Pressure and b factor
def pressure_vs_N(sigmas, N_values, L, simul_time, K_scale, replicas):
    Pressure = {sigma: [] for sigma in sigmas}
    b_factor = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for N in N_values:

            pres = []
            ke = []

            for _ in range(replicas):
                try:
                    sim = Simul(simul_time=simul_time, sigma=sigma, L=L, N=N, K_scale=K_scale)
                except ValueError:
                    pres.append(np.nan)
                    ke.append(np.nan)
                    continue

                P, Ec = sim.md_step()
                pres.append(P)
                ke.append(Ec)

            P_avg = np.nanmean(pres)
            ke_avg = np.nanmean(ke)

            Pressure[sigma].append(P_avg)

            if np.isnan(P_avg) or P_avg == 0:
                b_factor[sigma].append(np.nan)
            else:
                V = L * L
                b = (V / N) - (ke_avg / (N * P_avg))
                b_factor[sigma].append(b)

    return Pressure, b_factor


def pressure_vs_L(sigmas, L_values, N, simul_time, K_scale, replicas):
    Pressure = {sigma: [] for sigma in sigmas}
    b_factor = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for L in L_values:

            pres = []
            ke = []

            for _ in range(replicas):
                try:
                    sim = Simul(simul_time=simul_time, sigma=sigma, L=L, N=N, K_scale=K_scale)
                except ValueError:
                    pres.append(np.nan)
                    ke.append(np.nan)
                    continue

                P, Ec = sim.md_step()
                pres.append(P)
                ke.append(Ec)

            P_avg = np.nanmean(pres)
            ke_avg = np.nanmean(ke)

            Pressure[sigma].append(P_avg)

            if np.isnan(P_avg) or P_avg == 0:
                b_factor[sigma].append(np.nan)
            else:
                V = L * L
                b = (V / N) - (ke_avg / (N * P_avg))
                b_factor[sigma].append(b)

    return Pressure, b_factor


def pressure_vs_K(sigmas, L, N, simul_time, K_values, replicas):
    Pressure = {sigma: [] for sigma in sigmas}
    Energy = {sigma: [] for sigma in sigmas}
    b_factor = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for K_scale in K_values:

            pres = []
            ke = []

            for _ in range(replicas):
                try:
                    sim = Simul(simul_time=simul_time, sigma=sigma, L=L, N=N, K_scale=K_scale)
                except ValueError:
                    pres.append(np.nan)
                    ke.append(np.nan)
                    continue

                P, Ec = sim.md_step()
                pres.append(P)
                ke.append(Ec)

            P_avg = np.nanmean(pres)
            ke_avg = np.nanmean(ke)

            Pressure[sigma].append(P_avg)
            Energy[sigma].append(ke_avg)

            if np.isnan(P_avg) or P_avg == 0:
                b_factor[sigma].append(np.nan)
            else:
                V = L * L
                b = (V / N) - (ke_avg / (N * P_avg))
                b_factor[sigma].append(b)

    return Pressure, b_factor, Energy

#   PLOTTING FUNCTIONS

def plot_b_vs_N(sigmas, N_values, b_factor):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        bvals = np.array(b_factor[sigma])
        valid = ~np.isnan(bvals)
        plt.plot(1 / np.array(N_values)[valid], bvals[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("1/N")
    plt.ylabel("b (excluded volume parameter)")
    plt.title("b vs N for different σ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_b_vs_V(sigmas, L_values, b_factor):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        bvals = np.array(b_factor[sigma])
        valid = ~np.isnan(bvals)
        plt.plot(np.array(L_values)[valid]**2, bvals[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("V = L²")
    plt.ylabel("b (excluded volume parameter)")
    plt.title("b vs V for different σ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_b_vs_K(sigmas, K, b_factor):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        bvals = np.array(b_factor[sigma])
        valid = ~np.isnan(bvals)
        K_data = np.array(K[sigma])
        validk = ~np.isnan(K_data)
        plt.plot(np.array(K_data)[validk], bvals[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("K scale")
    plt.ylabel("b (excluded volume parameter)")
    plt.title("b vs K for different σ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pressure_vs_V(sigmas, L_values, results):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        data = np.array(results[sigma])
        valid = ~np.isnan(data)
        plt.plot(1 / (np.array(L_values)[valid]**2), data[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("1/V")
    plt.ylabel("Pressure P")
    plt.title("Pressure vs 1/V for different σ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pressure_vs_N(sigmas, N_values, results):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        data = np.array(results[sigma])
        valid = ~np.isnan(data)
        plt.plot(N_values, data[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("Number of particles N")
    plt.ylabel("Pressure P")
    plt.title("Pressure vs N for different σ")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_pressure_vs_K(sigmas, K, results):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        data = np.array(results[sigma])
        valid = ~np.isnan(data)
        K_data = np.array(K[sigma])
        validk = ~np.isnan(K_data)
        plt.plot(K_data[validk], data[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("K scale")
    plt.ylabel("Pressure P")
    plt.title("Pressure vs K for different σ")
    plt.grid(True)
    plt.legend()
    plt.show()



#  MAIN MENU

def main():

    simul_time = 10.0
    K_scale = 1.0
    replicas = 10

    # Define parameter ranges
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N_values = list(range(10, 160, 10))
    L_values = [4, 5, 6, 8, 10, 12, 15]
    K_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print("\nSelect the plot you want to generate:")
    print("1) N particles experiment")
    print("2) Box volume experiment")
    print("3) Kinetic energy experiment")
  
    choice = input("Enter your choice: ")

    if choice == "1":
        P, b = pressure_vs_N(sigmas, N_values, L=10, simul_time=simul_time, K_scale=K_scale, replicas=replicas)
        plot_pressure_vs_N(sigmas, N_values, P)
        plot_b_vs_N(sigmas, N_values, b)

    elif choice == "2":
        P, b = pressure_vs_L(sigmas, L_values, N=40, simul_time=simul_time, K_scale=K_scale, replicas=replicas)
        plot_pressure_vs_V(sigmas, L_values, P)
        plot_b_vs_V(sigmas, L_values, b)

    elif choice == "3":
        P, b, E = pressure_vs_K(sigmas, L=10, N=40, simul_time=simul_time, K_values=K_values, replicas=replicas)
        plot_pressure_vs_K(sigmas, E, P)
        plot_b_vs_K(sigmas, E, b)

    else:
        print("Invalid option. Exiting.")


if __name__ == "__main__":
    main()
