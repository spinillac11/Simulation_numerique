import numpy as np
import math
import random

class Simul:
    """ 
    This is the prototype of the simulation code
    It moves the particles with at _velocity, using a vector notation: numpy should be used.
    """
    def __init__(self, simul_time, sigma, L, N, K_scale):
        margen_extra = 0.001
        np.seterr(all='ignore')  # remove errors in where statements
        self.sigma = sigma  # particle radius
        self.N = N   
        self.simul_time = simul_time
        self.L = L
        self.K_scale = K_scale

        # Inicializar posiciones usando cuadrícula
        self.position = self._generar_posiciones_cuadricula(margen_extra)
        
        self.velocity = K_scale*np.random.normal(size=self.position.shape)  # random velocities
        self.l, self.m = np.triu_indices(self.position.shape[0], k=1)  # all pairs of indices between particles
    
    def _generar_posiciones_cuadricula(self, margen_extra):
        """
        Genera posiciones usando una cuadrícula para evitar superposición
        """
        # Calcular espaciamiento entre partículas
        espaciamiento = 2 * self.sigma + margen_extra
        
        # Calcular número máximo de partículas por dimensión
        num_por_lado = int(self.L // espaciamiento)
        total_posiciones = num_por_lado * num_por_lado
        
        if total_posiciones < self.N:
            raise ValueError(f"No hay suficientes posiciones en la cuadrícula. Máximo: {total_posiciones}, Solicitadas: {self.N}")
        
        # Generar todas las posiciones posibles en la cuadrícula
        todas_posiciones = []
        for i in range(num_por_lado):
            for j in range(num_por_lado):
                x = i * espaciamiento + self.sigma + margen_extra/2
                y = j * espaciamiento + self.sigma + margen_extra/2
                # Asegurar que estén dentro de los límites
                if x <= self.L - self.sigma and y <= self.L - self.sigma:
                    todas_posiciones.append([x, y])
        
        # Seleccionar N posiciones aleatorias
        if len(todas_posiciones) < self.N:
            raise ValueError(f"Solo se pudieron generar {len(todas_posiciones)} posiciones válidas")
        
        indices_seleccionados = random.sample(range(len(todas_posiciones)), self.N)
        posiciones_seleccionadas = [todas_posiciones[i] for i in indices_seleccionados]
        
        return np.array(posiciones_seleccionadas)

    def wall_time(self):
        positive_time = (self.L-self.sigma-self.position)/self.velocity
        neg_time = (self.sigma-self.position)/self.velocity
        collision_time = np.where(self.velocity >= 0, positive_time, neg_time)

        first_collision_time = np.min(collision_time)

        where = np.where(collision_time == first_collision_time)
        particle = where[0][0] #First index par
        direction = where[1][0] #Second index dir
        return first_collision_time, particle, direction

    def pair_time(self):
        delta_r = self.position[self.m] - self.position[self.l]   # relative positions
        delta_v = self.velocity[self.m] - self.velocity[self.l]   # relative velocities

        A = np.sum(delta_v * delta_v, axis=1)
        B = 2 * np.sum(delta_v * delta_r, axis=1)
        C = np.sum(delta_r * delta_r, axis=1) - (2 * self.sigma)**2

        Delta = B**2-4*A*C
        valid = (Delta > 0) & (B < 0)

        sqrt_Delta = np.sqrt(Delta)

        t_coll = np.where(valid, (-B-sqrt_Delta) / (2 * A), np.inf)

        first_collision_time = np.min(t_coll)

        idx = np.argmin(t_coll)

        return first_collision_time, self.l[idx], self.m[idx]
    
    def md_step(self):
        ke_start = (self.velocity**2).sum()/2.   # starting kinetic energy

        Delta_P = 0
        current_time = 0
        
        w_time, particle, direction = self.wall_time()
        p_time, particle_1, particle_2 = self.pair_time()

        time_min = min(w_time, p_time)
         
        while current_time + time_min < self.simul_time:

            if w_time < p_time:
                self.position += w_time * self.velocity
                self.velocity[particle, direction] = -self.velocity[particle, direction]
                current_time += w_time
                w_time, particle, direction = self.wall_time()
                p_time, particle_1, particle_2 = self.pair_time()
                time_min = min(w_time, p_time)
                Delta_P += 2*np.abs(self.velocity[particle, direction])
    
            else:
                self.position += p_time * self.velocity
                dR = self.position[particle_1]-self.position[particle_2]
                r = (self.position[particle_1]-self.position[particle_2])/np.sqrt(np.sum(dR*dR))
                dV = self.velocity[particle_1] - self.velocity[particle_2]
                self.velocity[particle_1] = self.velocity[particle_1] - r*(np.sum(r*dV))
                self.velocity[particle_2] = self.velocity[particle_2] + r*(np.sum(r*dV))
                current_time += p_time
                p_time, particle_1, particle_2 = self.pair_time()
                w_time, particle, direction = self.wall_time()
                time_min = min(w_time, p_time)

        self.position += (self.simul_time-current_time) * self.velocity

        assert math.isclose(ke_start,  (self.velocity**2).sum()/2.)  

        pressure = Delta_P / (4*self.L*self.simul_time)
        return pressure, ke_start

    def _str_(self):   # this is used to print the position and velocity of the particles
        p = np.array2string(self.position)
        v = np.array2string(self.velocity)
        return 'pos= '+p+'\n'+'vel= '+v+'\n'

    
# import numpy as np
# import math


# class Simul:
#     """ 
#     This is the prototype of the simulation code
#     It moves the particles with at _velocity, using a vector notation: numpy should be used.
#     """
#     def __init__(self, simul_time, sigma, L, N, K_scale):
#         np.seterr(all='ignore')  # remove errors in where statements
#         self.sigma = sigma  # particle radius
#         self.N = N   
#         self.simul_time = simul_time
#         self.L = L
#         self.K_scale = K_scale

#         self.position = (self.L-2*self.sigma)*np.random.rand(self.N, 2) + self.sigma # starting positions
#         self.velocity = K_scale*np.random.normal(size=self.position.shape)  # random velocities
#         self.l, self.m = np.triu_indices(self.position.shape[0], k=1)  # all pairs of indices between particles

    
        
#     def wall_time(self):
#         positive_time = (self.L-self.sigma-self.position)/self.velocity
#         neg_time = (self.sigma-self.position)/self.velocity
#         collision_time = np.where(self.velocity >= 0, positive_time, neg_time)

#         first_collision_time = np.min(collision_time)

#         where = np.where(collision_time == first_collision_time)
#         particle = where[0][0] #First index par
#         direction = where[1][0] #Second index dir
#         return first_collision_time, particle, direction

#     def pair_time(self):
#         delta_r = self.position[self.m] - self.position[self.l]   # relative positions
#         delta_v = self.velocity[self.m] - self.velocity[self.l]   # relative velocities

#         A = np.sum(delta_v * delta_v, axis=1)
#         B = 2 * np.sum(delta_v * delta_r, axis=1)
#         C = np.sum(delta_r * delta_r, axis=1) - (2 * self.sigma)**2

#         Delta = B**2-4*A*C
#         valid = (Delta > 0) & (B < 0)

#         sqrt_Delta = np.sqrt(Delta)

#         t_coll = np.where(valid, (-B-sqrt_Delta) / (2 * A), np.inf)

#         first_collision_time = np.min(t_coll)

#         idx = np.argmin(t_coll)

#         return first_collision_time, self.l[idx], self.m[idx]
    
#     def md_step(self):
#         ke_start = (self.velocity**2).sum()/2.   # starting kinetic energy

#         Delta_P = 0
#         current_time = 0
        
#         w_time, particle, direction = self.wall_time()
#         p_time, particle_1, particle_2 = self.pair_time()

#         time_min = min(w_time, p_time)
         
#         while current_time + time_min < self.simul_time:

#             if w_time < p_time:
#                 self.position += w_time * self.velocity
#                 self.velocity[particle, direction] = -self.velocity[particle, direction]
#                 current_time += w_time
#                 w_time, particle, direction = self.wall_time()
#                 p_time, particle_1, particle_2 = self.pair_time()
#                 time_min = min(w_time, p_time)
#                 Delta_P += 2*np.abs(self.velocity[particle, direction])
    
#             else:
#                 self.position += p_time * self.velocity
#                 dR = self.position[particle_1]-self.position[particle_2]
#                 r = (self.position[particle_1]-self.position[particle_2])/np.sqrt(np.sum(dR*dR))
#                 dV = self.velocity[particle_1] - self.velocity[particle_2]
#                 self.velocity[particle_1] = self.velocity[particle_1] - r*(np.sum(r*dV))
#                 self.velocity[particle_2] = self.velocity[particle_2] + r*(np.sum(r*dV))
#                 current_time += p_time
#                 p_time, particle_1, particle_2 = self.pair_time()
#                 w_time, particle, direction = self.wall_time()
#                 time_min = min(w_time, p_time)

#         self.position += (self.simul_time-current_time) * self.velocity

#         assert math.isclose(ke_start,  (self.velocity**2).sum()/2.)  
        
#         if self.position.any() > self.L or self.position.any() < 0:
#             raise ValueError("Particle outside the box")

#         pressure = Delta_P / (4*self.L*self.simul_time)
#         return pressure, ke_start

#     def __str__(self):   # this is used to print the position and velocity of the particles
#         p = np.array2string(self.position)
#         v = np.array2string(self.velocity)
#         return 'pos= '+p+'\n'+'vel= '+v+'\n'