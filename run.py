from simul import Simul
from animate import Animate
import matplotlib.pyplot as plt
import numpy as np


def main():
    P_list=[]
    np.random.seed()  # set random numbers to be always the same
    simulation = Simul(simul_time=0.06, sigma=0.1, L=10, N=100)  # sigma particle radius # L box size
    print(simulation.__doc__)  # print the documentation from the class
    
    pressure = 0
    for i in range(10000):
        p, k=simulation.md_step()
        pressure += p
        if i%100 == 0:
            P_list.append(pressure/(100*0.006))
            pressure = 0 
    
    #print(P_list)

    plt.plot(P_list)
    plt.show()

    animate = Animate(simulation)
    animate.go(nframes=400)  # number of animation steps
    print(simulation)  # print last configuration to screen


if __name__ == '__main__':
    main()
