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
# At room temperature a ratio of 1 electron in 100,000 copper atoms is really free.
#   Same as saying only 0.001% or 10^-5 of the conduction electrons are free.


import cupy as cp
import numpy as np   # numpy as np for CPU and now just for visualization 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.constants import e, epsilon_0, electron_mass, elementary_charge, c    
electron_charge=elementary_charge
coulombs_constant = 8.9875517873681764e9  # Coulomb's constant

import dask
from dask import delayed
from dask.distributed import Client, wait, LocalCluster
import multiprocessing

import os

gridx = 3300 # 
gridy = 10   # 
gridz = 10   # 

pulse_range=1000       # how many planes will be given pulse - we simulate half toward middle of this at each end

#     gridx gridy gridz  = total electrons
# Enough GPU memory
#        1500 20 20   = 600000 
#        3300 10 10   = 330000
#        800 40 40    = 1280000
#        400 50 50    = 1000000
#        300 100 100  = 3000000
# Not enough GPU memory
#       3500 10 10   = 350000
#       3200 20 20   = 1280000
#       2000 20 20   = 800000

# Initial electron speed 2,178,278 m/s
# electron_speed= 2178278  
electron_speed= 2188058


# Atom spacing in meters
hydrogen_spacing = 3.34e-9  # 3.34 nanometers between atoms in hydrogen gas
copper_spacing = 0.128e-9  # 3.34 nanometers between atoms in copper solid
initial_spacing = copper_spacing*47  # 47^3 is about 100,000 and 1 free electron for every 100,000 copper atoms
initial_radius = 5.29e-11 #  initial electron radius for hydrogen atom - got at least two times
pulse_offset =100*initial_spacing    #  how much the first few planes are offset
pulse_speed = 0    # in meters per second 
pulse_sinwave = False  # True if pulse should be sin wave
pulsehalf=False    # True to only pulse half the plane
initialize_orbits=False          # can have electrons initialized to moving if True and not moving if False

# bounds format is  ((minx,  maxx) , (miny, maxy), (minz, maxz))
bounds = ((0, gridx*initial_spacing), (0, gridy*initial_spacing), (0, gridz*initial_spacing))
# bounds = ((-1.0*initial_spacing, (gridx+1.0)*initial_spacing), (-1.0*initial_spacing, (gridy+1.0)*initial_spacing), (-1.0*initial_spacing, (gridz+1.0)*initial_spacing))

# Time stepping
num_steps =  400     # how many simulation steps
DisplaySteps = 10    # every so many simulation steps we call the visualize code
WireSteps = 1        # every so many simulation steps we call the visualize code
visualize_start= int(pulse_range/2) # have initial pulse electrons we don't really want to see 
visualize_stop = int(gridx-pulse_range/2) # really only goes up to one less than this but since starts at zero this many
visualize_plane_step = int((visualize_stop-visualize_start)/7) # Only show one every this many planes in data
speedup = 20       # sort of rushing the simulation time
proprange=visualize_stop-visualize_start # not simulating either end of the wire so only middle range for signal to propagage
dt = speedup*proprange*initial_spacing/c/num_steps  # would like total simulation time to be long enough for light wave to just cross grid 

coulombs_constant = 1 / (4 * cp.pi * epsilon_0)  # Coulomb's constant 

# Make string of some settings to put on output graph 
sim_settings = f"gridx {gridx} gridy {gridy} gridz {gridz} speedup {speedup} Spacing: {initial_spacing:.8e} PulseS {pulse_speed:.8e} PulseO {pulse_offset:.8e} Steps: {num_steps}"



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





# Initialize the 3 of the main arrays (forces ok at zeros)
def initialize_atoms():

    global initial_radius, electron_velocities, electron_positions, nucleus_positions, gridx, gridy, gridz, initial_spacing, initialize_orbits, electron_speed

    # Initialize nucleus positions
    x, y, z = cp.indices((gridx, gridy, gridz))
    nucleus_positions = cp.stack((x, y, z), axis=-1) * initial_spacing

    # Random angles for the initial positions
    theta = cp.random.uniform(0, 2*cp.pi, size=(gridx, gridy, gridz))
    phi = cp.random.uniform(0, cp.pi, size=(gridx, gridy, gridz))

    # Position in spherical coordinates
    ex = nucleus_positions[..., 0] + initial_radius * cp.sin(phi) * cp.cos(theta)
    ey = nucleus_positions[..., 1] + initial_radius * cp.sin(phi) * cp.sin(theta)
    ez = nucleus_positions[..., 2] + initial_radius * cp.cos(phi)

    electron_positions = cp.stack((ex, ey, ez), axis=-1)

    if initialize_orbits:
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
    global gridx, gridy, gridz, bounds, nucleus_positions, electron_speed, electron_velocities  # all these global are really constants
    global visualize_start, visualize_stop, visualize_plane_step

    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111, projection='3d')
    # ax.clear()  Think not needed anymore
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    minx = visualize_start*initial_spacing
    maxx = visualize_stop*initial_spacing
    ax.set_xlim(minx,maxx)   # set display bounds which are different than simulation bounds
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])


   # formatter = ScalarFormatter(useOffset=False, useMathText=True)
   # ax.xaxis.set_major_formatter(formatter)
   # ax.yaxis.set_major_formatter(formatter)
   # ax.zaxis.set_major_formatter(formatter)


    # Example: setting ticks for the X-axis
    x_ticks = np.linspace(minx, maxx, num=7) # Adjust 'num' for the number of ticks
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:.2e}" for tick in x_ticks]) # Formatting to scientific notation

    y_ticks = np.linspace(bounds[1][0], bounds[1][1], num=7) # Adjust 'num' for the number of ticks
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:.2e}" for tick in y_ticks]) # Formatting to scientific notation

    z_ticks = np.linspace(bounds[2][0], bounds[2][1], num=7) # Adjust 'num' for the number of ticks
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f"{tick:.2e}" for tick in z_ticks]) # Formatting to scientific notation

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
    ax.set_title(f"Step {step} Time: {t:.8e} {sim_settings}")

    # Use os.path.join to create the file path
    filename = os.path.join('simulation', f'step_{step}.png')
    plt.savefig(filename)

    print("minxd  =",minxd)
    print("maxxd  =",maxxd)
    print("mins  =",mins)
    print("maxs  =",maxs)

    del epositions, evelocities     # we don't need copy here any more - telling garbage collector
    plt.close(fig)  # Close the figure to free memory



# The wire runs in the X direction and electrons at each grid X,Y,Z 
# have an x,y,z position that started near nucleus x,y,z
# Want to know how far the average in each X slice of the wire has moved in the x direction
# as electrical signal in simulation should be moving in that direction
def calculate_wire(epositions):
    global gridx, gridy, gridz, nucleus_positions

    # Calculate the difference in the x-direction
    xdiff = epositions[:,:,:,0] - nucleus_positions[:,:,:,0]

    # Calculate the total difference for each x slice
    totalxdiff = cp.sum(xdiff, axis=(1, 2))  # Summing over y and z axes

    # Calculate the average difference for each x slice
    averaged_xdiff = totalxdiff / (gridy * gridz)

    return(averaged_xdiff)

def visualize_wire(averaged_xdiff, step, t):
    # Plotting
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    ax.plot(cp.asnumpy(averaged_xdiff), marker='o')  # Convert to NumPy array for plotting

    ax.set_xlabel('X index')
    ax.set_ylabel('Average X Difference')
    ax.set_title(f'Step {step} Time: {t:.8e} sec {sim_settings}')
    ax.grid(True)

    # Save the figure
    filename = os.path.join('simulation', f'wire_{step}.png')
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory


# Need to make Initial pulse by moving few rows of electrons 
#
# Displace electrons in half the first 3 x layers
# transverse wave so in x=0 plane we displace in the x direction
def pulse():
    global pulse_range, pulse_speed, electron_positions, electron_velocities
    yrange=gridy
    if pulsehalf:
        yrange=int(gridy/2)
    for x in range(0,pulse_range):
        if pulse_sinwave:
            offset_add = -1*pulse_offset*np.sin(2*np.pi*x/pulse_range)
            speed_add = -1*pulse_speed*np.sin(2*np.pi*x/pulse_range)
        else:
            offset_add = pulse_offset
            speed_add = pulse_speed
        for y in range(yrange):
            for z in range(gridz):
                electron_positions[x,y,z][0] += offset_add
                electron_velocities[x,y,z][0] += speed_add





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
    global electron_velocities, electron_positions, bounds, forces, electron_mass, visualize_start, visualize_stop

    # acceleration based ono F=ma
    acceleration = forces / electron_mass

    # Update velocities
    new_velocities = electron_velocities + acceleration * dt

    # Update positions using vectors
    new_positions = electron_positions + new_velocities * dt


    # Create a mask for X indices that should be updated
    # Generate an array representing the X indices
    x_indices = cp.arange(gridx).reshape(gridx, 1, 1, 1)  # Reshape for broadcasting
    # Create a boolean mask where True indicates the indices to be updated
    # We only update the same part we are visualizing 
    update_mask = (x_indices > visualize_start) & (x_indices < visualize_stop)
    # update_mask = x_indices > -1 

    # Apply updates using the mask for selective application
    electron_velocities = cp.where(update_mask, new_velocities, electron_velocities)
    electron_positions = cp.where(update_mask, new_positions, electron_positions)

    # keep things in bounds
    for i, (min_bound, max_bound) in enumerate(bounds):
        # Check for upper boundary
        over_max = new_positions[..., i] > max_bound
        electron_positions[..., i][over_max] = max_bound
        electron_velocities[..., i][over_max] *= -1

        # Check for lower boundary
        below_min = electron_positions[..., i] < min_bound
        electron_positions[..., i][below_min] = min_bound
        electron_velocities[..., i][below_min] *= -1

    print("Max positions:", cp.max(cp.linalg.norm(electron_positions, axis=1)))
    print("Max velocity:", cp.max(cp.linalg.norm(electron_velocities, axis=1)))


def main():
    global gridx, gridy, gridz, initial_spacing, num_steps, speedup, forces, electron_positions, electron_velocities, dt

    print("In main")
    checkgpu()
    initialize_atoms()
    os.makedirs('simulation', exist_ok=True) # Ensure the simulation directory exists


    # Create a LocalCluster with a custom death timeout and then a Client
    cluster = LocalCluster(n_workers=24, death_timeout='600s')
    client= Client(cluster)
    futures = []

    print("Doing first visualization")
    copypositions=electron_positions.copy()
    copyvelocities=electron_velocities.copy()
    future = client.submit(visualize_atoms, copypositions, copyvelocities, -1, 0.0)
    futures.append(future)
    del copypositions, copyvelocities     # we don't need copy here any more - telling garbage collector
    print("Doing pulse")
    pulse()
    # main simulation loop
    for step in range(num_steps):
        t = step * dt
        print("In main", step)
        if step % WireSteps == 0:
            WireStatus=calculate_wire(electron_positions)
            future = client.submit(visualize_wire, WireStatus, step, t)
            futures.append(future)
            del WireStatus
        if step % DisplaySteps == 0:
            print("Display", step)
            copypositions=electron_positions.copy()
            copyvelocities=electron_velocities.copy()
            future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t)
            futures.append(future)
            del copypositions, copyvelocities     # we don't need copy here any more - telling garbage collector

        print("Updating force", step)
        forces=calculate_forces()

        print("Updating position and velocity", t)
        update_pv(dt)
        cp.cuda.Stream.null.synchronize()         # free memory on the GPU

    copypositions=electron_positions.copy()
    copyvelocities=electron_velocities.copy()
    future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t) # If we end at 200 we need last output
    futures.append(future)
    del copypositions, copyvelocities     # we don't need copy here any more - telling garbage collector
    wait(futures)



if __name__ == '__main__':
    main()
