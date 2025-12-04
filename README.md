## 📝 README: 2D Gas Molecular Dynamics Simulation

This project uses **Molecular Dynamics (MD)** to simulate a two-dimensional gas of hard particles and study its pressure ($P$) properties in relation to the **Number of Particles ($N$)**, the **Volume ($V$ - box area)**, and the **Kinetic Energy ($E_c$)**, which is proportional to the **Temperature ($T$)**.

The objective is to compare the simulation results with the **Ideal Gas Law** ($P \times V = N k_B T$) and the **Van der Waals equation** ($P \times (V/N - b) = k_B T$), and to attempt to evaluate the value of the excluded volume parameter ($b$) of Van der Waals.
---
### ▶️ How to Run the Code

The code is divided into two main files:

1.  **`simul.py`**: Contains the `Simul` class, which implements the Molecular Dynamics algorithm for hard particles, calculating collisions (between particles and with the walls) and evolving the system. It returns the pressure and total kinetic energy.
2.  **`run.py`**: Contains the functions to execute multiple simulations, compute averages, derive the $b$ parameter, and generate the result plots. By runing this file you will be asked to chosse wich varible will change to study the pressure in the sistem. If you chosse to vary the number of particles the code will take about 25 minutes tu run beacuse of the hevy simulation for to many particles (if you want results faster reduce the simul_time and the amount of replicas).


