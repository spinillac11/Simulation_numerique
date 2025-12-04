import numpy as np
import matplotlib.pyplot as plt
from simul import Simul
from scipy.stats import linregress
import os

os.makedirs("figures", exist_ok=True)

def execute_simulation(sigma, L, N, K_scale, simul_time, replicas):
    pres = [] # to store pressure results per replica
    ke = [] # to store kinetic energy results per replica
    for _ in range(replicas):
        try:
            sim = Simul(simul_time=simul_time, sigma=sigma, L=L, N=N, K_scale=K_scale)
        except ValueError: # Not enough space to place particles without overlap
            pres.append(np.nan)
            ke.append(np.nan)
            continue

        P, Ec = sim.md_step()
        pres.append(P)
        ke.append(Ec)
    
    P_avg = np.nanmean(pres)
    ke_avg = np.nanmean(ke)

    return P_avg, ke_avg

# Experimental functions to compute Pressure and b factor
def pressure_vs_N(sigmas, N_values, L, simul_time, K_scale, replicas):
    Pressure = {sigma: [] for sigma in sigmas}
    Energy = {sigma: [] for sigma in sigmas}
    b_factor = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for N in N_values:

            P_avg, ke_avg = execute_simulation(sigma, L, N, K_scale, simul_time, replicas)

            Pressure[sigma].append(P_avg)
            Energy[sigma].append(ke_avg)

            if np.isnan(P_avg) or P_avg == 0: # to avoid division by zero
                b_factor[sigma].append(np.nan)
            else:
                V = L * L
                b = (V / N) - (ke_avg / (N * P_avg))
                b_factor[sigma].append(b)

    return Pressure, b_factor, Energy


def pressure_vs_L(sigmas, L_values, N, simul_time, K_scale, replicas):
    Pressure = {sigma: [] for sigma in sigmas}
    Energy = {sigma: [] for sigma in sigmas}
    b_factor = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for L in L_values:
            
            P_avg, ke_avg = execute_simulation(sigma, L, N, K_scale, simul_time, replicas)

            Pressure[sigma].append(P_avg)
            Energy[sigma].append(ke_avg)

            if np.isnan(P_avg) or P_avg == 0:
                b_factor[sigma].append(np.nan) # to avoid division by zero
            else:
                V = L * L
                b = (V / N) - (ke_avg / (N * P_avg))
                b_factor[sigma].append(b)

    return Pressure, b_factor, Energy


def pressure_vs_K(sigmas, L, N, simul_time, K_values, replicas):
    Pressure = {sigma: [] for sigma in sigmas}
    Energy = {sigma: [] for sigma in sigmas}
    b_factor = {sigma: [] for sigma in sigmas}

    for sigma in sigmas:
        for K_scale in K_values:

            P_avg, ke_avg = execute_simulation(sigma, L, N, K_scale, simul_time, replicas)

            Pressure[sigma].append(P_avg)
            Energy[sigma].append(ke_avg)

            if np.isnan(P_avg) or P_avg == 0:
                b_factor[sigma].append(np.nan) # to avoid division by zero
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
    plt.ylabel("b (paramètre de volume exclu)")
    plt.title("b en fonction de N pour différents σ (L = 10, K_scale = 1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/b_vs_N.png", dpi=300)
    plt.show()


def plot_b_vs_V(sigmas, L_values, b_factor):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        bvals = np.array(b_factor[sigma])
        valid = ~np.isnan(bvals)
        plt.plot(np.array(L_values)[valid]**2, bvals[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("V = L²")
    plt.ylabel("b (paramètre de volume exclu)")
    plt.title("b en fonction du volume pour différents σ (N = 40, K_scale = 1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/b_vs_V.png", dpi=300)
    plt.show()

def plot_b_vs_K(sigmas, K, b_factor, b_regression):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        bvals = np.array(b_factor[sigma])
        K_data = np.array(K[sigma])
        valid = (~np.isnan(bvals)) & (~np.isnan(K_data))

        plt.plot(K_data[valid], bvals[valid], "-o", label=f"sigma={sigma} data")

        if sigma in b_regression:
            b_reg, m, r = b_regression[sigma]
            plt.hlines(b_reg, xmin=np.nanmin(K_data[valid]), xmax=np.nanmax(K_data[valid]), 
                       linestyles="--", label=f"sigma={sigma} regression b={b_reg:.3f}")

    plt.xlabel("Énergie cinétique K")
    plt.ylabel("b (paramètre de volume exclu)")
    plt.title("b en fonction de K pour différents σ (N = 40, L = 10)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/b_vs_K.png", dpi=300)
    plt.show()


def plot_pressure_vs_V(sigmas, L_values, P_results):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        data = np.array(P_results[sigma])
        valid = ~np.isnan(data)
        plt.plot(1 / (np.array(L_values)[valid]**2), data[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("1/V")
    plt.ylabel("Pression P")
    plt.title("Pression en fonction de 1/V pour différents σ (N = 40, K_scale = 1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/pressure_vs_V.png", dpi=300)
    plt.show()


def plot_pressure_vs_N(sigmas, N_values, P_results):
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        data = np.array(P_results[sigma])
        valid = ~np.isnan(data)
        plt.plot(np.array(N_values)[valid], data[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("Nombre de particules N")
    plt.ylabel("Pression P")
    plt.title("Pression en fonction de N pour différents σ (L = 10, K_scale = 1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/pressure_vs_N.png", dpi=300)
    plt.show()


def plot_pressure_vs_K(sigmas, K, P_results, N, L):

    plt.figure(figsize=(10, 6))

    b_regression = {}
    V = L * L

    print("linear regression results for P vs K:")
    for sigma in sigmas:
        data = np.array(P_results[sigma])
        valid = ~np.isnan(data)
        K_data = np.array(K[sigma])
        validk = ~np.isnan(K_data)

        if np.sum(valid & validk) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(K_data[validk], data[valid])
            if slope is not None:
                b_reg = (V / N) - (1 / (slope * N))
            else:
                b_reg = np.nan

            b_regression[sigma] = (b_reg, slope, r_value**2)
            print(f"sigma={sigma}: slope={slope}, intercept={intercept}, R²={r_value**2}")

        plt.plot(K_data[validk], data[valid], "-o", label=f"sigma={sigma}")

    plt.xlabel("Kinnetic energy K")
    plt.ylabel("Pressure P")
    plt.title("Pressure vs K for different σ (N = 40, L = 10)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/pressure_vs_K.png", dpi=300)
    plt.show()

    return b_regression # return regression results for b factor in b vs K plot

#  MAIN MENU

def main():
    # Simul time and replicas to get averages
    simul_time = 10
    replicas = 10

    # Fixed parameters
    L = 10
    N = 40
    K_scale = 1.0

    # Define parameter ranges for experiments
    sigmas = [0.005, 0.008, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N_values = list(range(10, 160, 10))
    L_values = [4, 5, 6, 8, 10, 12, 15]
    K_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # User menu for selecting experiment
    print("\n Selecto parameter to vary:")
    print("1) N particles experiment")
    print("2) Box volume experiment")
    print("3) Kinetic energy experiment")
  
    choice = input("Enter your choice: ")

    # Pressure vs N experiment
    if choice == "1":
        P, b, E = pressure_vs_N(sigmas, N_values, L=L, simul_time=simul_time, K_scale=K_scale, replicas=replicas)
        plot_pressure_vs_N(sigmas, N_values, P)
        plot_b_vs_N(sigmas, N_values, b)
    # Pressure vs L experiment
    elif choice == "2":
        P, b, E = pressure_vs_L(sigmas, L_values, N=N, simul_time=simul_time, K_scale=K_scale, replicas=replicas)
        plot_pressure_vs_V(sigmas, L_values, P)
        plot_b_vs_V(sigmas, L_values, b)
    # Pressure vs K experiment
    elif choice == "3":
        P, b, E = pressure_vs_K(sigmas, L=L, N=N, simul_time=simul_time, K_values=K_values, replicas=replicas)
        b_reg = plot_pressure_vs_K(sigmas, E, P, N, L)
        plot_b_vs_K(sigmas, E, b, b_reg)

    else:
        print("Invalid option. Exiting.")


if __name__ == "__main__":
    main()
