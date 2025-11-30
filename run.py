import numpy as np
import matplotlib.pyplot as plt
from simul import Simul

# -------------------------------
# Utility functions
# -------------------------------

def pressure_vs_N(L, sigma_min, sigma_max, simul_time, Ns, steps, K_scale):
    Pressures = []
    Energies = []
    bs = []

    for N in Ns:
        sim = Simul(simul_time=simul_time, sigma_min=sigma_min, sigma_max=sigma_max, L=L, N=N, K_scale=K_scale)

        b = np.pi * np.mean(sim.sigma ** 2)
        bs.append(b)

        P_accum = 0
        E_accum = 0

        for _ in range(steps):
            p, k = sim.md_step()
            P_accum += p
            E_accum += k

        Pressures.append(P_accum / steps)
        Energies.append(E_accum / steps)

    return np.array(Pressures), np.array(Energies), np.array(bs)


def plot_ideal_gas(P, Ec, V, Ns):
    ratio = P * V / Ec
    plt.figure()
    plt.plot(Ns, ratio, marker='o')
    plt.xlabel("Number of particles N")
    plt.ylabel("PV / Ec")
    plt.title("Ideal Gas Test: PV/Ec = constant?")
    plt.grid()
    plt.show()


def plot_van_der_waals(P, Ec, V, Ns, b):
    Ns = np.array(Ns)       
    b = np.array(b)
    ratio = P * (V - Ns * b) / Ec
    plt.figure()
    plt.plot(Ns, ratio, marker='o')
    plt.xlabel("Number of particles N")
    plt.ylabel("P (V/N - b) / Ec")
    plt.title("Van der Waals Test: P(V/N-b)/Ec = constant?")
    plt.grid()
    plt.show()


def main():
    print("Select a simulation to run:")
    print("1) Ideal Gas Test (very small particles)")
    print("2) Van der Waals Test (large particles)")

    choice = input("Enter 1, or 2: ")
    #choice2 = input("Enter 1, 2, or 3:")

    L = 10
    simul_time = 0.5
    Ns = [4, 8, 10, 12, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160]
    steps = 200
    K_scale = 2
    
    if choice == "1":
        print("\nRunning Ideal Gas Test...")
        P, Ec, b = pressure_vs_N(L=L, sigma_min=0.01, sigma_max=0.01, simul_time=simul_time, Ns=Ns, steps=steps, K_scale=K_scale)
        V = L * L
        plot_ideal_gas(P, Ec, V, Ns)

    elif choice == "2":
        print("\nRunning Van der Waals Test...")
        P, Ec, b = pressure_vs_N(L=L, sigma_min=0.2, sigma_max=0.2, simul_time=simul_time, Ns=Ns, steps=steps, K_scale=K_scale)
        V = L * L
        plot_van_der_waals(P, Ec, V, Ns, b)

    else:
        print("Invalid option.")
        return


if __name__ == "__main__":
    main()
