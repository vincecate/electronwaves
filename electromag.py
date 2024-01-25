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
# The calculated radii of the orbits for the hydrogen atom in the Bohr model for principal quantum numbers
# 
# n=1,2,3 are as follows:
# 
# For n=1 (ground state): The radius is approximately  5.29×10 −11 meters (or 52.9 picometers).
# For n=2: The radius is approximately 2.12×10 −10 meters (or 212 picometers).
# For n=3: The radius is approximately 4.76×10 −10 meters (or 476 picometers).
#
# Visible light falls within a range of wavelengths from approximately 380 nanometers (nm) to about 750 nm. 
#    About 100 times the distance between air molecules!
#
# In the case of a large plate (where 'large' means that the plate's dimensions are much larger than the distance from the plate), 
# the electric field is approximately uniform close to the plate. This means that the field strength does not significantly decrease 
# for short distances (like centimeters) from the plate's center or surface. The uniformity of the field holds true as 
# long as the distance from the plate is small compared to the dimensions of the plate.
# 
# The diameter of a 22-gauge wire is approximately 2,510,234 copper atoms wide. 
#   Electrons in a wire are propagating a plane wave!  The field is pushing on
#   electrons thousands of atoms in front at the same time, so the wave can 
#   move far faster than any electron moves!
#
# Where the Voyager probes are there is about 1 atom per cubic cm.
# If a plane wave is miles wide, that could be enough to propagate.
#
# https://en.wikipedia.org/wiki/Hydraulic_analogy  - water and electricity similarity - but speed of light is plane wave!
#
# https://www.youtube.com/watch?v=sDlZ-aY9GN4   - a magnetic field in one frame can be an electric field in another -relativity
# https://www.youtube.com/watch?v=1TKSfAkWWN0   - veritasium on relativity and electric vs magnetic
# https://www.youtube.com/watch?v=Ii7rgIQawko   - The Science Asylum on relativity and manetic vs electric
#     There are so many positive and negative charges that a slight length change in one can make significant net charge
#
# Updated plan Jan 24, 2024
#    Simulate electron as cloud around nucleus with spring model - kind of like Maxwell model
#    With a plane the net electric field does not really drop off.  This is why light is so fast!
#
# Would be amazing result of this theory if it could correctly show:
#  1- speed of light comes out right from plane wave and electrons charge and mass
#  2- light of certain frequencies can drive electrons off as in experiments
#  3- magnetic field from moving charges - if we can simulate moving charges curving in mag field using instant force beteen electrons
#  4 - ampiers law that two wires next to each othe with current going the same way attract and opposite then repel
#
# Updated plan Jan 25, 2024
#     Ignor nucleus and think of electrons in a wire.
#         If get to edge of wire need to reflect back - have square wire :-)
#         For first simulation can start all with zero velocity

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
usedask=True

#grid_size = 40   # 30 can be down to 2 mins for 10 dt if all goes well
gridx = 20   # To start only simulating few layers 
gridy = 40   # 30 can be down to 2 mins for 10 dt if all goes well
gridz = 40   # 30 can be down to 2 mins for 10 dt if all goes well


# Declare global variables
global fig, ax
# Initial electron speed 2,178,278 m/s
# electron_speed= 2178278  
electron_speed= 2188058


# Atom spacing in meters
atom_spacing = 3.34e-9  # 3.34 nanometers between atoms
nearby_grid = 40    # we are only calculating forces from atoms this grid distance in each direction 
initial_radius = 5.29e-11 #  initial electron radius for hydrogen atom - got at least two times
pulse_perc = 0.5 # really a ratio to initial radius but think like percent
pulserange=5       # 0 to 4 will be given pulse
simrange=pulserange-1   # we don't bother simulating the pulse electrons
pulsehalf=False    # True to only pulse half the plane
maxy=(gridy+1)*atom_spacing  #  in wire can't go outside wire
maxz=(gridz+1)*atom_spacing  #  in wire can't go outside wire
miny=-1                      # edge of wire
minz=-1                      # edge of wire
minx=-1                      # edge of wire

# Time stepping
num_steps =  200
DisplaySteps = 1     # every so many simulation steps we call the visualize code
visualize_start= simrange # really the 3rd plane since starts at 0
visualize_stop = gridx-3 # really only goes up to one less than this but since starts at zero this many
speedup = 10       # sort of rushing the simulation time

coulombs_constant = 1 / (4 * cp.pi * epsilon_0)  # Coulomb's constant 



nucleus_positions = cp.zeros((gridx, gridy, gridz, 3))
electron_positions = cp.zeros((gridx, gridy, gridz, 3))
electron_velocities = cp.zeros((gridx, gridy, gridz, 3))
forces = cp.zeros((gridx, gridy, gridz, 3))

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

    # Ensure the simulation directory exists
    os.makedirs('simulation', exist_ok=True)

def clear_visualization():
    global ax
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


# New idea is to show the average electron Y position relative to nucleus in each plane x=0, x=1, ...
# If there is a wave where Y is changing this would show it
# To break into parts for different cores we need to
#    combine min/max results
#    probably not add things to ax.scatter inside different cores
# XXX this is slow and serial and would be good to make faster somehow
def visualize_atoms(step, t):
    global gridx, gridy, gridz, electron_positions, nucleus_positions, electron_speed, electron_velocities
    global visualize_start, visualize_stop

    minxd = 10  # find electron with minimum Y distance from local nucleus
    maxxd = 0   # find electron with maximum Y distance from local nucleus
    mins = 10*electron_speed  # find minimum speed of an electron
    maxs = 0   # find maximum speed of an electron


    # Clear the previous plot
    clear_visualization()  

    for x in range(visualize_start, visualize_stop):  # pulse at x=0,1,2  so first light will be 3
        totalxdiff=0.0
        for y in range(gridy):
            for z in range(gridz):
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
                xdiff = electron_pos[0] - nucleus_pos[0]  # How far in X direction from nucleus 
                totalxdiff += xdiff
                distance = abs(xdiff)   # absolute value of distance
                # Normalize the distance and map to color - 
                # adding 1.5x should get us to 0.5 to 2.5 when still circular and 0 to 3 if bit wild
                normalized_distance =  abs((distance - 0.5*initial_radius)/initial_radius)
                color = plt.cm.viridis(normalized_distance)
                ax.scatter(*electron_pos, color=color,  s=10)
                if(normalized_distance<minxd):
                    minxd=normalized_distance
                if(normalized_distance>maxxd):
                    maxxd=normalized_distance
                speed = cp.linalg.norm(electron_velocities[x,y,z]);
                if(speed<mins):
                    mins=speed
                if(speed>maxs):
                    maxs=speed
        print(" ", totalxdiff / (gridy * gridz), ",", end='')
    # Set title with the current time in the simulation
    ax.set_title(f"Step {step}  Simulation Time: {t} seconds")

    # Use os.path.join to create the file path
    filename = os.path.join('simulation', f'step_{step}.png')
    plt.savefig(filename)
    if (guion):
         plt.show()
    # Process GUI events and wait briefly to keep the GUI responsive
    #plt.pause(0.001)

    print("minxd  =",minxd)
    print("maxxd  =",maxxd)
    print("mins  =",mins)
    print("maxs  =",maxs)




# Global 3D arrays with 3 unit arrays as data
# nucleus_positions 
# electron_positions 
# electron_velocities 
# forces
# Initialize the 3 of the main arrays (forces ok at zeros)
def initialize_atoms():
    for x in range(gridx):
        for y in range(gridy):
            for z in range(gridz):
                nucleus_positions[x, y, z] = cp.array([x, y, z]) * atom_spacing
                initialize_electron(x, y, z)   # both position and velocity of electron XXX debug

# Need to make Initial pulse by moving few rows of electrons XXX
#
# Displace electrons in half the first 3 x layers
# transverse wave so in x=0 plane we displace in the x direction
def pulse():
    global pulserange
    displacement = pulse_perc * initial_radius
    yrange=gridy
    if pulsehalf:
        yrange=int(gridy/2)
    for x in range(0,pulserange):
        for y in range(yrange):
            for z in range(gridz):
                electron_positions[x,y,z][0] += displacement




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
                    if 0 <= nx < gridx and 0 <= ny < gridy and 0 <= nz < gridz:
                        nearby_nucleus_position = nucleus_positions[nx, ny, nz]
                        nearby_electron_position = electron_positions[nx, ny, nz]
                        #   add force from nearby nucleus 
                        # totalforce += calculate_coulomb_force(electron_position, nearby_nucleus_position, -e, e)
                        #   add force from nearby electron 
                        totalforce += calculate_coulomb_force(electron_position, nearby_electron_position, -e, -e)
        return(totalforce)

# enough parallel work to pass off to a different core
def update_onepart(x, dt):
    global gridy, gridz
    for y in range(gridy):
        for z in range(gridz):
            acceleration = forces[x, y, z] / m_e
            electron_velocities[x, y, z] += acceleration * dt
            electron_positions[x, y, z] += electron_velocities[x, y, z] * dt
            #  These 4 ifs are to bounce electron off 4 sides of our square wire
            # If out of bounds and headed away reverse that part of velocity vector
            if electron_positions[x, y, z][1] > maxy and electron_velocities[x, y, z][1] > 0:
                electron_velocities[x, y, z][1] = -electron_velocities[x, y, z][1]              # bounce off maxy
            if electron_positions[x, y, z][2] > maxz and electron_velocities[x, y, z][2] > 0:
                electron_velocities[x, y, z][2] = -electron_velocities[x, y, z][2]              # bounce off maxz
            if electron_positions[x, y, z][1] < miny and electron_velocities[x, y, z][1] < 0:
                electron_velocities[x, y, z][1] = -electron_velocities[x, y, z][1]              # bounce off miny
            if electron_positions[x, y, z][2] < minz and electron_velocities[x, y, z][2] < 0:
                electron_velocities[x, y, z][2] = -electron_velocities[x, y, z][2]              # bounce off minz
            if electron_positions[x, y, z][0] < minx and electron_velocities[x, y, z][0] < 0:
                electron_velocities[x, y, z][0] = -electron_velocities[x, y, z][0]              # bounce off minx

def main():
    global gridx, gridy, gridz, atom_spacing, num_steps, plt, speedup

    checkgpu()
    initialize_atoms()
    initialize_visualization()
    visualize_atoms(-1, 0)       # want one visualize right before pulse and one right after - mod 0 is 0 after ok
    pulse()

    client = Client(n_workers=24)  # You can adjust the number of workers
    # main simulation loop
    dt = speedup*gridx*atom_spacing/c/num_steps  # would like total simulation time to be long enough for light wave to just cross grid 
    for step in range(num_steps):
        t = step * dt

        if step % DisplaySteps == 0:
            print("Display", step)
            visualize_atoms(step, t)
        #plt.pause(0.01)

        print("Updating force", step)
        if usedask:
            # Create delayed tasks for each iteration of the loop to compute force
            tasks = []
            # Create tasks for each grid position
            for x in range(simrange, gridx):      # not simulating pulse electrons - like held by battery
                for y in range(gridy):
                    for z in range(gridz):
                        task = delayed(compute_force)(x, y, z)
                        tasks.append(task)

            # Compute the tasks in parallel
            forces_results = dask.compute(*tasks, scheduler='processes')  # can be processes or syncronous for debug
            #forces_results = dask.compute(*tasks, scheduler='threads')  # can be processes or syncronous for debug

            # Update the global forces array directly
            idx = 0
            for x in range(simrange, gridx):
                for y in range(gridy):
                    for z in range(gridz):
                        forces[x, y, z] = forces_results[idx]
                        idx += 1
        else:
            for x in range(simrange, gridx):
                for y in range(gridy):
                    compute_onepart(x,y)

        print("Updating position and velocity", t)
        if usedask and False:
            tasks = []
            for x in range(simrange, gridx):
                task = delayed(update_onepart)(x, dt)
                tasks.append(task)
            results = dask.compute(*tasks, scheduler='processes') # Use Dask to compute the tasks in parallel
        else:
            for x in range(gridx):
                update_onepart(x,dt)

    visualize_atoms(step, t)  # If we end at 200 we need last output




if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
