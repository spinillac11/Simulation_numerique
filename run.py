import numpy as np
import matplotlib.pyplot as plt
from simul import Simul


def run_experiment(variable, values, sigma, simul_time, steps):
    Pressures = []
    Energies = []

    for val in values:

        # Default parameters
        N = 60
        L = 10
        K_scale = 1.0

        # Modify depending on what we vary
        if variable == "N":
            N = val
        elif variable == "L":
            L = val
        elif variable == "K":
            K_scale = val

        simulation = Simul(simul_time=simul_time,sigma=sigma, L=L, N=N, K_scale=K_scale)

        P_accum = 0
        E_accum = 0

        for _ in range(steps):
            p, k = simulation.md_step()
            P_accum += p
            E_accum += k

        Pressures.append(P_accum / steps)
        Energies.append(E_accum / steps)

    return np.array(Pressures), np.array(Energies)



def main():

    print("\nSelect what variable to vary:")
    print("1) Number of particles N")
    print("2) Box size L")
    print("3) Initial kinetic energy scale K")

    var_choice = input("Enter 1, 2, or 3: ").strip()

    if var_choice == "1":
        variable = "N"
        values = [10, 20, 40, 60, 80, 100, 120, 150]
    elif var_choice == "2":
        variable = "L"
        values = [6, 8, 10, 12, 14, 16]
    elif var_choice == "3":
        variable = "K"
        values = [0.5, 1.0, 2.0, 3.0]
    else:
        print("Invalid option.")
        return
    
    N = 60
    L = 10
    K_scale = 1.0

    simul_time = 0.2
    steps = 300

    sigma = 0.8

    P, Ec = run_experiment(variable, values, sigma, simul_time, steps)


    if variable == "L":
        Ls = np.array(values)
        plt.figure()
        plt.plot(1/ (Ls * Ls), P, marker='o')
        plt.xlabel("Volume")
        plt.ylabel("Presure")
        plt.title("Ideal Gas: P vs V")
        plt.grid()
        plt.savefig("")

    if variable == "N":
        N_vals = np.array(values)
        plt.figure()
        plt.plot(N_vals, P, marker='o')
        plt.xlabel("N particles")
        plt.ylabel("Presure")
        plt.title("Ideal Gas: P vs N")
        plt.grid()
        plt.show()

    if variable == "K":
        plt.figure()
        plt.plot(Ec, P, marker='o')
        plt.xlabel("Kinetic Energy")
        plt.ylabel("Presure")
        plt.title("Ideal Gas: P vs K")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()