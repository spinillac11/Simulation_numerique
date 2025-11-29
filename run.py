from simul import Simul
from animate import Animate
import matplotlib.pyplot as plt
import numpy as np


def main():
    P_list=[]
    K_list=[]
    np.random.seed()  # set random numbers to be always the same
    simulation = Simul(simul_time=0.006, sigma_min=0.1, sigma_max=0.2, L=10, N=100) 
    print(simulation.__doc__)  # print the documentation from the class
    
    pressure = 0
    kinetic = 0
    for i in range(5000):
        p, k=simulation.md_step()
        pressure += p
        kinetic += k
        if i%100 == 0:
            P_list.append(pressure/(100*0.006))
            K_list.append(kinetic/100)
            pressure = 0 
            kinetic = 0
    
    #print(P_list)

    plt.plot(P_list)
    plt.show()

    animate = Animate(simulation)
    animate.go(nframes=400)  # number of animation steps
    #print(simulation)  # print last configuration to screen


if __name__ == '__main__':
    main()
