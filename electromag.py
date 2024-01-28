#!/usr/bin/python3
# Requires:
# pip install numpy
# pip install cupy-cuda115
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

use_gpu = True  # True means Cupy and GPY, False means NumPy and CPU

if use_gpu:
    import cupy as cp
else:
    import numpy as cp

import numpy as np   # numpy as np for CPU and now just for visualization 
import matplotlib.pyplot as plt
from scipy.constants import e, epsilon_0, electron_mass, elementary_charge, m_e, c    # m_e is mass of electrong 
electron_charge=elementary_charge
coulombs_constant = 8.9875517873681764e9  # Coulomb's constant

import dask
from dask import delayed
from dask.distributed import Client, wait
import multiprocessing

import os

guion=False

#grid_size = 40   # 30 can be down to 2 mins for 10 dt if all goes well
gridx = 100   # To start only simulating few layers 
gridy = 50   # 30 can be down to 2 mins for 10 dt if all goes well
gridz = 50   # 30 can be down to 2 mins for 10 dt if all goes well

# With numpy 80,40,40 and simstop 40 is about 1 per minute

# Declare global variables
global fig, ax
# Initial electron speed 2,178,278 m/s
# electron_speed= 2178278  
electron_speed= 2188058


# Atom spacing in meters
# atom_spacing = 3.34e-9  # 3.34 nanometers between atoms in hydrogen gas
atom_spacing = 0.128e-9  # 3.34 nanometers between atoms in copper solid
initial_radius = 5.29e-11 #  initial electron radius for hydrogen atom - got at least two times
pulse_perc = 0.5 # really a ratio to initial radius but think like percent
pulserange=5       # 0 to 4 will be given pulse
simxstart=pulserange-1   # we don't bother simulating the pulse electrons
simxstop=int(gridx/2)        # want wire in front not to move or suck electrons by not being there
pulsehalf=False    # True to only pulse half the plane
einitialmoving=False          # can have electrons initialized to moving if True and not moving if False

# bounds format is  ((minx,  maxx) , (miny, maxy), (minz, maxz))
bounds = ((-1.0*atom_spacing, (gridx+1)*atom_spacing), (-1.0*atom_spacing, (gridy+1)*atom_spacing), (-1.0*atom_spacing, (gridz+1)*atom_spacing))

# Time stepping
num_steps =  200
DisplaySteps = 1     # every so many simulation steps we call the visualize code
visualize_plane_step = 10 # Only show every 3rd plane
visualize_start= simxstart # really the 3rd plane since starts at 0
visualize_stop = simxstop # really only goes up to one less than this but since starts at zero this many
speedup = 1       # sort of rushing the simulation time

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




# Global 3D arrays with 3 unit arrays as data
# nucleus_positions 
# electron_positions 
# electron_velocities 
# forces
# With numpy we had a class HydrogenAtom but no more with cupy.

import cupy as cp

# Initialize the 3 of the main arrays (forces ok at zeros)
def initialize_atoms():
    global initial_radius, electron_velocities, electron_positions, nucleus_positions, gridx, gridy, gridz, atom_spacing, einitialmoving, electron_speed

    # Initialize nucleus positions
    x, y, z = cp.indices((gridx, gridy, gridz))
    nucleus_positions = cp.stack((x, y, z), axis=-1) * atom_spacing

    # Random angles for the initial positions
    theta = cp.random.uniform(0, 2*cp.pi, size=(gridx, gridy, gridz))
    phi = cp.random.uniform(0, cp.pi, size=(gridx, gridy, gridz))

    # Position in spherical coordinates
    ex = nucleus_positions[..., 0] + initial_radius * cp.sin(phi) * cp.cos(theta)
    ey = nucleus_positions[..., 1] + initial_radius * cp.sin(phi) * cp.sin(theta)
    ez = nucleus_positions[..., 2] + initial_radius * cp.cos(phi)

    electron_positions = cp.stack((ex, ey, ez), axis=-1)

    if einitialmoving:
        # Random vectors
        random_vectors = cp.random.random((gridx, gridy, gridz, 3))

        # Electron vectors
        electron_vectors = electron_positions - nucleus_positions

        # Perpendicular vectors
        perpendicular_vectors = cp.cross(electron_vectors, random_vectors)

        # Normalized vectors
        norms = cp.linalg.norm(perpendicular_vectors, axis=-1, keepdims=True)
        normalized_vectors = perpendicular_vectors / norms

        electron_velocities = normalized_vectors * electron_speed
    else:
        electron_velocities = cp.zeros((gridx, gridy, gridz, 3))




#  Want to make visualization something we can hand off to a dask core to work on
#   so we will put together something and hand it off 
#   with 12 cores we can do well
# For now nucleus_positions is a constant - doing electrons in wire first
def visualize_atoms(epositions, evelocities, step, t):
    global gridx, gridy, gridz, nucleus_positions, electron_speed, electron_velocities  # all these global are really constants
    global visualize_start, visualize_stop, visualize_plane_step

    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    minxd = 10  # find electron with minimum Y distance from local nucleus
    maxxd = 0   # find electron with maximum Y distance from local nucleus
    mins = 10*electron_speed  # find minimum speed of an electron
    maxs = 0   # find maximum speed of an electron


    #  Could do safety check here and not include exlectrons out of bounds in visualization XXX
    for x in range(visualize_start, visualize_stop, visualize_plane_step):  # pulse at x=0,1,2  so first light will be 3
        totalxdiff=0.0
        for y in range(gridy):
            for z in range(gridz):
                # we sometimes test without cupy by doing "import numpy as cp" 
                if 'cupy' in str(type(epositions[x,y,z])):
                    electron_pos = cp.asnumpy(epositions[x,y,z])
                else:
                    electron_pos = epositions[x,y,z]
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
                speed = cp.linalg.norm(evelocities[x,y,z]);
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

    print("minxd  =",minxd)
    print("maxxd  =",maxxd)
    print("mins  =",mins)
    print("maxs  =",maxs)





# Need to make Initial pulse by moving few rows of electrons 
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


def calculate_forces():
    global electron_positions, forces, coulombs_constant, electron_charge

    # Calculate pairwise differences in position (broadcasting)
    delta_r = electron_positions[:, cp.newaxis, :] - electron_positions[cp.newaxis, :, :]

    # Calculate distances and handle division by zero
    distances = cp.linalg.norm(delta_r, axis=-1)
    distances[cp.eye(distances.shape[0], dtype=bool)] = cp.inf

    # Calculate forces (Coulomb's Law)
    force_magnitude = coulombs_constant * (electron_charge ** 2) / distances**2

    # Normalize force vectors and multiply by magnitude
    normforces = force_magnitude[..., cp.newaxis] * delta_r / distances[..., cp.newaxis]

    print("Mean force magnitude:", cp.mean(cp.linalg.norm(normforces, axis=1)))
    print("Max force magnitude:", cp.max(cp.linalg.norm(normforces, axis=1)))

    # Sum forces from all other electrons for each electron
    return(cp.sum(normforces, axis=1))




def update_pv(dt):
    global electron_velocities, electron_positions, bounds, forces, m_e

    # Update velocities based on acceleration (F = ma)
    acceleration = forces / m_e
    new_velocities = electron_velocities + acceleration * dt

    # Update positions using vectors
    new_positions = electron_positions + electron_velocities * dt

    for i, (min_bound, max_bound) in enumerate(bounds):
        # Check for upper boundary
        over_max = new_positions[..., i] > max_bound
        new_positions[..., i][over_max] = max_bound
        new_velocities[..., i][over_max] *= -1

        # Check for lower boundary
        below_min = new_positions[..., i] < min_bound
        new_positions[..., i][below_min] = min_bound
        new_velocities[..., i][below_min] *= -1

    print("Max positions:", cp.max(cp.linalg.norm(new_positions, axis=1)))
    print("Max velocity:", cp.max(cp.linalg.norm(new_velocities, axis=1)))
    return(new_positions, new_velocities)


def main():
    global gridx, gridy, gridz, atom_spacing, num_steps, plt, speedup, forces, electron_positions, electron_velocities

    print("In main")
    checkgpu()
    initialize_atoms()
    os.makedirs('simulation', exist_ok=True) # Ensure the simulation directory exists

    client= Client(n_workers=24)
    futures = []

    print("Doing first visualization")
    copypositions=electron_positions.copy()
    copyvelocities=electron_velocities.copy()
    future = client.submit(visualize_atoms, copypositions, copyvelocities, -1, 0.0)
    futures.append(future)
    print("Doing pulse")
    pulse()
    # main simulation loop
    dt = speedup*simxstop*atom_spacing/c/num_steps  # would like total simulation time to be long enough for light wave to just cross grid 
    for step in range(num_steps):
        t = step * dt
        print("In main", step)

        if step % DisplaySteps == 0:
            print("Display", step)
            copypositions=electron_positions.copy()
            copyvelocities=electron_velocities.copy()
            future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t)
            futures.append(future)

        print("Updating force", step)
        forces=calculate_forces()

        print("Updating position and velocity", t)
        electron_positions,electron_velocities=update_pv(dt)

    copypositions=electron_positions.copy()
    copyvelocities=electron_velocities.copy()
    future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t) # If we end at 200 we need last output
    futures.append(future)
    wait(futures)



if __name__ == '__main__':
    main()
