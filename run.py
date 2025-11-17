from simul import Simul
from animate import Animate
import numpy as np

def main():
    P_list=[]
    np.random.seed()  # set random numbers to be always the same
    simulation = Simul(simul_time=0.006, sigma=0.1, L=10, N=100)  # sigma particle radius # L box size
    print(simulation.__doc__)  # print the documentation from the class
    # for i in range(5):
    #     p=simulation.md_step()  #  read the result of the pressure from a single time step
    #     P_list.append(p) 

    animate = Animate(simulation)
    animate.go(nframes=400)  # number of animation steps
    print(simulation)  # print last configuration to screen


if __name__ == '__main__':
    main()
