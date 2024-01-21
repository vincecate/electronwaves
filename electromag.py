#!/usr/bin/python3
# Requires:
# pip install numpy
# pip install cupy
# pip install matplotlib
# pip install scipy
# pip install dask
# pip install distributed
#
#  We want to simulate an input wave and see if it can propagate through 
#  electons in hydrogen gas.
#  For now we just move some of the electrons in a layer of atoms on one edge and see what happens.
#
#
#*
# * Air molecules are around 0.1 nanometers
# * The distance between air molecules is around 3.34 nanometers
# * So about 33.38 times larger distance 
# *
# * In the Bohr model of the hydrogen atom, the electron orbits the nucleus in a manner similar to planets orbiting the sun. While this model is a simplification and
# doesn't fully represent the complexities of quantum mechanics, it provides a useful approximation for certain calculations.
#
#According to the Bohr model, the speed of an electron in a hydrogen atom can be calculated using the formula:
#
#v=e^2/(2*ϵ0*h) * n
#
#
#Where:
#
#    v is the speed of the electron.
#    e is the elementary charge (1.602×10−191.602×10−19 Coulombs).
#    ϵ0 the vacuum permittivity (8.854×10−128.854×10−12 C22/Nm22).
#    h is the Planck constant (6.626×10−346.626×10−34 Js).
#    n is the principal quantum number (for the first orbit of hydrogen, n=1n=1).
#
#Let's calculate this speed for the hydrogen atom in its ground state (where n=1n=1).
#
#In the Bohr model of the hydrogen atom, the speed of an electron in its ground state (with the principal quantum number n=1n=1) 
# is approximately 2,187,278 meters per second.   This is about 0.73% the speed of light.

import numpy as cp    # cupy for GPU
import numpy as np   # numpy as np for CPU and now just for visualization 
import matplotlib.pyplot as plt
from scipy.constants import e, epsilon_0, m_e, c
import dask
from dask import delayed
from dask.distributed import Client
import dask
import multiprocessing
import os

guion=False
usedask=False

grid_size = 15   # 30 can be down to 2 mins for 10 dt if all goes well

visualize_start= 2 # really the 3rd plane since starts at 0
visualize_stop = 4 # really only goes up to one less than this but since starts at zero this many

# Declare global variables
global fig, ax
# Initial electron speed 2,178,278 m/s
# electron_speed= 2178278  
electron_speed= 2188058


# Atom spacing in meters
atom_spacing = 3.34e-9  # 3.34 nanometers between atoms
nearby_grid = 3    # we are only calculating forces from atoms this grid distance in each direction 
initial_radius = 5.29e-11 #  initial electron radius for hydrogen atom - got at least two times
pulse_perc = 0.2 # really a ratio to initial radius but think like percent

# Time stepping
num_steps =  200
DisplaySteps = 10     # every so many simulation steps we call the visualize code


coulombs_constant = 1 / (4 * cp.pi * epsilon_0)  # Coulomb's constant 



nucleus_positions = cp.zeros((grid_size, grid_size, grid_size, 3))
electron_positions = cp.zeros((grid_size, grid_size, grid_size, 3))
electron_velocities = cp.zeros((grid_size, grid_size, grid_size, 3))
forces = cp.zeros((grid_size, grid_size, grid_size, 3))

# Global 3D arrays with 3 unit arrays as data
# nucleus_positions 
# electron_positions 
# electron_velocities 
# forces
# Further initialization and computations go here


# Check if 'cp' is CuPy and CUDA is available
def checkgpu():
    if hasattr(cp, 'cuda') and cp.cuda.is_available():
        print("CUDA is available")
        num_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"Number of available GPUs: {num_gpus}")
    
        # Get the ID of the current GPU
        current_gpu_id = cp.cuda.get_device_id()
        print(f"Current GPU ID: {current_gpu_id}")
    else:
        print("CUDA is not available or cp is NumPy")



def calculate_coulomb_force(charge1_position, charge2_position, charge1, charge2):
    global coulombs_constant
    """
    Calculates the Coulomb force vector exerted on charge1 by charge2.

    Args:
    - charge1_position (cp.array): The position vector of the first charge.
    - charge2_position (cp.array): The position vector of the second charge.
    - charge1 (float): The magnitude of the first charge.
    - charge2 (float): The magnitude of the second charge.

    Returns:
    - cp.array: The force vector exerted on charge1 by charge2.
    """
    r_vector = charge1_position - charge2_position
    r = cp.linalg.norm(r_vector)
    if r != 0:
        f_magnitude = coulombs_constant * (charge1 * charge2) / (r ** 2)
        force_vector = f_magnitude * (r_vector / r)
        return force_vector
    else:
        return cp.array([0.0, 0.0, 0.0])




# Global 3D arrays with 3 unit arrays as data
# nucleus_positions 
# electron_positions 
# electron_velocities 
# forces
# With numpy we had a class HydrogenAtom but no more with cupy.

def initialize_electron(x, y, z):
    global initial_radius, electron_velocities, electron_positions, nucleus_positions

    # Initial electron radius 
    radius = initial_radius
    nucleus_position = nucleus_positions[x, y, z]

    # Random angle for the initial position
    theta = cp.random.uniform(0, 2*cp.pi)
    phi = cp.random.uniform(0, cp.pi)

    # Position in spherical coordinates
    ex = nucleus_position[0] + radius * cp.sin(phi) * cp.cos(theta)
    ey = nucleus_position[1] + radius * cp.sin(phi) * cp.sin(theta)
    ez = nucleus_position[2] + radius * cp.cos(phi)

    electron_positions[x, y, z] = cp.array([ex, ey, ez])

    # position done now velcoity
    # Random direction perpendicular to the vector from nucleus to electron
    electron_vector = electron_positions[x, y, z] - nucleus_positions[x, y, z]
    random_vector = cp.random.random(3)  # Random vector
    perpendicular_vector = cp.cross(electron_vector, random_vector)  # Cross product to ensure perpendicularity
    normalized_vector = perpendicular_vector / cp.linalg.norm(perpendicular_vector)

    electron_velocities[x, y, z] = normalized_vector * electron_speed



# Global 3D arrays with 3 unit arrays as data
# nucleus_positions 
# electron_positions 
# electron_velocities 
# forces

#Want to make one window for all visualization steps - just update each loop



def initialize_visualization():
    global fig, ax  # Declare as global
    fig = plt.figure(figsize=(12.8, 9.6))
    plt.ion()  # non-blocking
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Ensure the simulation directory exists
    os.makedirs('simulation', exist_ok=True)



# To break into parts for different cores we need to
#    combine min/max results
#    probably not add things to ax.scatter inside different cores
# XXX this is slow and serial and would be good to make faster somehow
def visualize_atoms(step, t):
    global grid_size, electron_positions, nucleus_positions, electron_speed, electron_velocities
    global visualize_start, visualize_stop

    mind = 10  # find electron with minimum distance from local nucleus
    maxd = 0   # find electron with maximum distance from local nucleus
    mins = 10*electron_speed  # find minimum speed of an electron
    maxs = 0   # find maximum speed of an electron


    # Clear the previous plot
    ax.clear()  

    for x in range(visualize_start, visualize_stop):  # pulse at x=0 so 0 and 1 are first intersting things 
        for y in range(grid_size):
            for z in range(grid_size):
                # we sometimes test without cupy by doing "import numpy as cp" 
                if 'cupy' in str(type(electron_positions[x,y,z])):
                    electron_pos = cp.asnumpy(electron_positions[x,y,z])
                else:
                    electron_pos = electron_positions[x,y,z]
                if 'cupy' in str(type(nucleus_positions[x,y,z])):
                    nucleus_pos = cp.asnumpy(nucleus_positions[x,y,z])
                else:
                    nucleus_pos = nucleus_positions[x,y,z]
                # wave is displacement in y plane so show that displacement with color
                distance = np.linalg.norm(electron_pos[1] - nucleus_pos[1])
                # Normalize the distance and map to color - trying to get normalized to be between 0 and 1
                # adding 1.5x should get us to 0.5 to 2.5 when still circular and 0 to 3 if bit wild
                normalized_distance =  (distance+(1.5*initial_radius))/(3*initial_radius) 
                color = plt.cm.viridis(normalized_distance)
                ax.scatter(*electron_pos, color=color,  s=10)
                if(normalized_distance<mind):
                    mind=normalized_distance
                if(normalized_distance>maxd):
                    maxd=normalized_distance
                speed = cp.linalg.norm(electron_velocities[x,y,z]);
                if(speed<mins):
                    mins=speed
                if(speed>maxs):
                    maxs=speed

    # Set title with the current time in the simulation
    ax.set_title(f"Step {step}  Simulation Time: {t} seconds")

    # Use os.path.join to create the file path
    filename = os.path.join('simulation', f'step_{step}.png')
    plt.savefig(filename)
    if (guion):
         plt.show()
    # Process GUI events and wait briefly to keep the GUI responsive
    plt.pause(0.001)

    print("mind  =",mind)
    print("maxd  =",maxd)
    print("mins  =",mins)
    print("maxs  =",maxs)




# Global 3D arrays with 3 unit arrays as data
# nucleus_positions 
# electron_positions 
# electron_velocities 
# forces
# Initialize the 3 of the main arrays (forces ok at zeros)
def initialize_atoms():
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                nucleus_positions[x, y, z] = cp.array([x, y, z]) * atom_spacing
                initialize_electron(x, y, z)   # both position and velocity of electron

# Need to make Initial pulse by moving few rows of electrons XXX
#
# Displace electrons in the first 3 x layers
# transverse wave so in x=0 plane we displace in the y direction
def pulse():
    displacement = pulse_perc * initial_radius
    for y in range(grid_size):
        for z in range(grid_size):
            for x in range(0,3):
                electron_positions[x,y,z][1] += displacement




# Global 3D arrays with 3 unit arrays as data
# nucleus_positions 
# electron_positions 
# electron_velocities 
# forces

# need to look at nearby atoms
def compute_force(x, y, z):
        electron_position=electron_positions[x, y, z]
        nucleus_position=nucleus_positions[x, y, z]
        totalforce = cp.array([0.0, 0.0, 0.0])
        # Iterate over nearby atoms within  3 grid units
        for nx in range(x-nearby_grid, x+nearby_grid):
            for ny in range(y-nearby_grid, y+nearby_grid):
                for nz in range(z-nearby_grid, z+nearby_grid):
                    if nx == x and ny == y and nz == z:
                        # add force for own nucleus
                        totalforce += calculate_coulomb_force(electron_position, nucleus_position, -e, e)
                        continue  # If on own then no nearby to do

                    # Check if the position is within the grid
                    if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                        nearby_nucleus_position = nucleus_positions[nx, ny, nz]
                        nearby_electron_position = electron_positions[nx, ny, nz]
                        #   add force from nearby nucleus 
                        totalforce += calculate_coulomb_force(electron_position, nearby_nucleus_position, -e, e)
                        #   add force from nearby electron 
                        totalforce += calculate_coulomb_force(electron_position, nearby_electron_position, -e, -e)
        forces[x, y, z]=totalforce

# enough parallel work to pass off to a different core
def compute_onepart(x,y):
    global grid_size
    for z in range(grid_size):
           compute_force(x, y, z)
    return(0)

# enough parallel work to pass off to a different core
def update_onepart(x, dt):
    global grid_size
    for y in range(grid_size):
        for z in range(grid_size):
            acceleration = forces[x, y, z] / m_e
            electron_velocities[x, y, z] += acceleration * dt
            electron_positions[x, y, z] += electron_velocities[x, y, z] * dt

def main():
    global grid_size, atom_spacing, num_steps, plt

    checkgpu()
    initialize_visualization()
    initialize_atoms()
    pulse()

    client = Client(n_workers=24)  # You can adjust the number of workers
    # main simulation loop
    dt = grid_size*atom_spacing/c/num_steps  # would like total simulation time to be long enough for light wave to just cross grid 
    for step in range(num_steps):
        t = step * dt

        if step % DisplaySteps == 0:
            print("Display")
            visualize_atoms(step, t)
        #plt.pause(0.01)

        print("Updating force", step)
        if usedask:
            # Create delayed tasks for each iteration of the loop to compute force
            tasks = []
            for x in range(grid_size):
                for y in range(grid_size):
                        task = delayed(compute_onepart)(x,y)
                        tasks.append(task)
            results = dask.compute(*tasks, scheduler='processes') # Use Dask to compute the tasks in parallel
        else:
            for x in range(grid_size):
                for y in range(grid_size):
                    compute_onepart(x,y)

        print("Updating position and velocity", t)
        if usedask:
            tasks = []
            for x in range(grid_size):
                task = delayed(update_onepart)(x, dt)
                tasks.append(task)
            results = dask.compute(*tasks, scheduler='processes') # Use Dask to compute the tasks in parallel
        else:
            for x in range(grid_size):
                update_onepart(x,dt)

    visualize_atoms(step, t)  # If we end at 200 we need last output




if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
