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

#  Update idea Feb 9, 2024
# In classical simulations, introducing a "hard sphere" collision model for very close distances could prevent physical 
#  impossibilities like electron overlap. This involves detecting when electrons are within a certain minimum distance of 
# each other and then calculating their trajectories post-collision based on conservation of momentum and energy, similar to billiard balls colliding.
#
#  In the wire display it may be good to show average movement of each dX in a graph.  Perhaps this will show
#  a much faster wave than what is probably a and "electron density wave".   Feb 12, 2024


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
import sys

# Check if at least one argument is provided (excluding the script name)
if len(sys.argv) > 1:
    simnum = int(sys.argv[1])  # The first argument passed to the script
    print(f"Simulation number provided: {simnum}")
else:
    print("No simulation number provided. Exiting.")
    sys.exit(1)  # Exit the script with an error code


# $1 argument gets saved to simnum so can do batch of several simulations from script
# have a set to try to get projection to speed of light in full size plane wave

effective_electron_mass = electron_mass   #  default is the same
electron_thermal_speed = 1.1e6            # meters per second
bounce_distance = 1e-10                   # closer than this and we make electrons bounce off each other

# Making wider wires have deeper pulses so scaling is 3D to give better estimate for real wire extrapolation
if simnum==0:            # 
    gridx = 200          # 
    gridy = 10           # 
    gridz = 10           # 
    speedup = 1          # sort of rushing the simulation time
    pulse_width=80       # how many planes will be given pulse - we simulate half toward middle of this at each end
    num_steps =  2000    # how many simulation steps - note dt slows down as this gets bigger unless you adjust speedup

if simnum==1:            # 
    gridx = 2000          # 
    gridy = 10           # 
    gridz = 10           # 
    speedup = 300        # sort of rushing the simulation time
    pulse_width=800       # how many planes will be given pulse - we simulate half toward middle of this at each end
    num_steps =  5000    # how many simulation steps - note dt slows down as this gets bigger unless you adjust speedup

if simnum==2:            # 
    gridx = 70           #  could do 1500 before 2D code - here total is 28,000 electrons
    gridy = 20           # 
    gridz = 20           # 
    speedup = 100        # sort of rushing the simulation time
    pulse_width=40      # how many planes will be given pulse - we simulate half toward middle of this at each end
    num_steps =  2000    # how many simulation steps - note dt slows down as this gets bigger unless you adjust speedup

if simnum==3:            # Ug came out slower when was predicting much faster - might need to run longer to get real speed
    gridx = 30           # 
    gridy = 30           # 
    gridz = 30           # 
    speedup = 300        # sort of rushing the simulation time
    pulse_width=20      # how many planes will be given pulse - we simulate half toward middle of this at each end
    num_steps =  2000    # how many simulation steps - note dt slows down as this gets bigger unless you adjust speedup


if simnum==4:            #
    gridx = 400          # 
    gridy = 80           # 
    gridz = 80           # 
    speedup = 300        # sort of rushing the simulation time
    pulse_width=200      # how many planes will be given pulse - we simulate half toward middle of this at each end
    num_steps =  2000    # how many simulation steps - note dt slows down as this gets bigger unless you adjust speedup

if simnum==5:            #
    gridx = 160          # 
    gridy = 160          # 
    gridz = 160          # 
    speedup = 300        # sort of rushing the simulation time
    pulse_width=160      # Really want twice this but may be able to learn something with this.  
    num_steps =  2000    # how many simulation steps - note dt slows down as this gets bigger unless you adjust speedup

DisplaySteps = 5000  # every so many simulation steps we call the visualize code
WireSteps = 1        # every so many simulation steps we call the visualize code


#     gridx gridy gridz  = total electrons
# Enough GPU memory
#        1500 20 20   = 600000 
#        3300 10 10   = 330000
#        800 40 40    = 1280000
#        400 50 50    = 1000000
#        300 100 100  = 3000000

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
initial_spacing_gemini = 2.27e-10  # Gemini Ultra number for free electron spacing
initial_spacing = copper_spacing*47  # 47^3 is about 100,000 and 1 free electron for every 100,000 copper atoms
initial_radius = 5.29e-11 #  initial electron radius for hydrogen atom - got at least two times
pulse_sinwave = False  # True if pulse should be sin wave
pulsehalf=False    # True to only pulse half the plane

initialize_orbits=True          # can have electrons initialized to moving if True and not moving if False

# bounds format is  ((minx,  maxx) , (miny, maxy), (minz, maxz))
bounds = ((0, gridx*initial_spacing), (0, gridy*initial_spacing), (0, gridz*initial_spacing))
# bounds = ((-1.0*initial_spacing, (gridx+1.0)*initial_spacing), (-1.0*initial_spacing, (gridy+1.0)*initial_spacing), (-1.0*initial_spacing, (gridz+1.0)*initial_spacing))

# Time stepping
visualize_start= int(pulse_width/3) # have initial pulse electrons we don't really want to see 
visualize_stop = int(gridx-pulse_width/3) # really only goes up to one less than this but since starts at zero this many
visualize_plane_step = int((visualize_stop-visualize_start)/7) # Only show one every this many planes in data
sim_start = 0         # can be visualize_start
sim_stop = gridx      # can be visualize_stop
wire_start = 0        # can look at a smaller section 
wire_stop = gridx
max_neighbor_grids=50      # maximum up or down the X direction that we calculate forces for electrons

proprange=visualize_stop-visualize_start # not simulating either end of the wire so only middle range for signal to propagage
dt = speedup*proprange*initial_spacing/c/num_steps  # would like total simulation time to be long enough for light wave to just cross grid 

coulombs_constant = 1 / (4 * cp.pi * epsilon_0)  # Coulomb's constant 

# Make string of some settings to put on output graph 
sim_settings = f"simnum {simnum} gridx {gridx} gridy {gridy} gridz {gridz} speedup {speedup} \n Spacing: {initial_spacing:.8e} Pulse Width {pulse_width} ElcSpeed {electron_thermal_speed:.8e} Steps: {num_steps} dt: {dt:.8e}"

def GPUMem():
    # Get total and free memory in bytes
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()

    print(f"Total GPU Memory: {total_mem / 1e9} GB")
    print(f"Free GPU Memory: {free_mem / 1e9} GB")

GPUMem()

# grid_size is the total number of electrons
grid_size = gridx * gridy * gridz
# Initialize positions and velocities as single-dimensional arrays
#electron_positions = cp.zeros((gridx, gridy, gridz, 3))    # old way

electron_positions = cp.zeros((grid_size, 3))


electron_velocities = cp.zeros((grid_size, 3))
forces = cp.zeros((grid_size, 3))


def calculate_collision_velocity(v1, v2, p1, p2):
    """
    Calculate the new velocities of two particles undergoing an elastic collision.

    Args:
        v1 (ndarray): Velocity of the first particle before collision.
        v2 (ndarray): Velocity of the second particle before collision.
        p1 (ndarray): Position of the first particle.
        p2 (ndarray): Position of the second particle.

    Returns:
        Tuple[ndarray, ndarray]: New velocities of the first and second particles.
    """
    # Calculate the unit vector in the direction of the collision
    collision_vector = p1 - p2
    collision_vector /= cp.linalg.norm(collision_vector)

    # Calculate the projections of the velocities onto the collision vector
    v1_proj = cp.dot(v1, collision_vector)
    v2_proj = cp.dot(v2, collision_vector)

    # Swap the velocity components along the collision vector (elastic collision)
    v1_new = v1 - v1_proj * collision_vector + v2_proj * collision_vector
    v2_new = v2 - v2_proj * collision_vector + v1_proj * collision_vector

    return v1_new, v2_new





def detect_collisions(electron_positions, bounce_distance):
    n_electrons = electron_positions.shape[0]
    diff = electron_positions[:, None, :] - electron_positions[None, :, :]  # Shape: (n, n, 3)
    distances = cp.sqrt(cp.sum(diff**2, axis=2))  # Shape: (n, n)
    colliding = distances < bounce_distance
    cp.fill_diagonal(colliding, False)  # Ignore self-collision
    return colliding


#  This is just swapping velocities which is not accurate
def resolve_collisions(electron_positions, electron_velocities, bounce_distance):
    colliding = detect_collisions(electron_positions, bounce_distance)
    n_electrons = electron_positions.shape[0]

    for i in range(n_electrons):
        for j in range(i + 1, n_electrons):  # Avoid double processing pairs
            if colliding[i, j]:
                # Simplified collision resolution: Swap velocities
                electron_velocities[i], electron_velocities[j] = electron_velocities[j], electron_velocities[i]


def generate_thermal_velocities(num_electrons, temperature=300):
    """
    Generates random thermal velocity vectors for a given number of electrons
    at a specified temperature. The Maxwell-Boltzmann distribution is used to
    determine the magnitudes of the velocities.

    Args:
        num_electrons (int): Number of electrons to generate velocities for.
        temperature (float): Temperature in Kelvin.

    Returns:
        cupy.ndarray: An array of shape (num_electrons, 3) containing random
                      velocity vectors for each electron.
    """

    kb = 1.380649e-23  # Boltzmann constant, in J/K
    electron_mass = 9.1093837e-31  # Electron mass, in kg

    # Calculate the standard deviation of the speed distribution
    sigma = cp.sqrt(kb * temperature / electron_mass)

    # Generate random speeds from a Maxwell-Boltzmann distribution
    # Use the fact that the Maxwell-Boltzmann distribution for speeds in one dimension
    # is a normal distribution with mean 0 and standard deviation sigma
    vx = cp.random.normal(loc=0, scale=sigma, size=num_electrons)
    vy = cp.random.normal(loc=0, scale=sigma, size=num_electrons)
    vz = cp.random.normal(loc=0, scale=sigma, size=num_electrons)

    # Combine the velocity components into a single 2D array
    velocities = cp.stack((vx, vy, vz), axis=-1)

    return velocities



    # Optional: Plot a histogram for analysis
    # plt.hist(speeds)
    # filename = os.path.join('simulation', f'maxwell.boltzmann.png')
    # plt.savefig(filename)
    # plt.close()           # Close the figure to free memory



def initialize_atoms():
    global initial_radius, electron_velocities, electron_positions
    global initial_spacing, initialize_orbits, electron_speed, pulse_width, electron_thermal_speed

    grid_size = gridx * gridy * gridz  # Total number of atoms

    # Calculate the modified x positions with pulse spacing
    pulse_spacing = initial_spacing / 2
    grid_non_pulse = gridx - pulse_width
    total_space = initial_spacing * gridx
    rest_space = total_space - pulse_spacing * pulse_width
    rest_spacing = rest_space / grid_non_pulse
    half_spacing_x_positions = cp.linspace(0, (pulse_width - 1) * pulse_spacing, pulse_width)
    full_spacing_start = half_spacing_x_positions[-1] + rest_spacing
    full_spacing_x_positions = cp.linspace(full_spacing_start, total_space - rest_spacing, gridx - pulse_width)
    modified_x_positions = cp.concatenate((half_spacing_x_positions, full_spacing_x_positions))

    # Generate y and z positions using initial_spacing, ensuring the entire grid is covered
    y_positions = cp.arange(gridy) * initial_spacing
    z_positions = cp.arange(gridz) * initial_spacing

    # Create the full 3D grid of positions in a 2D format
    x_grid, y_grid, z_grid = cp.meshgrid(modified_x_positions, y_positions, z_positions, indexing='ij')
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()

    # Stack x, y, z positions to form the (grid_size, 3) electron_positions array
    electron_positions = cp.stack((x_flat, y_flat, z_flat), axis=-1)

    # Initialize velocities
    if initialize_orbits:
        electron_velocities = generate_thermal_velocities(grid_size, 300.0)  # Adjusted to generate velocities for all electrons
    else:
        electron_velocities = cp.zeros((grid_size, 3))  # Use the new 2D structure directly





#  Want to make visualization something we can hand off to a dask core to work on
#   so we will put together something and hand it off 
#   with 12 cores we can do well
# For now nucleus_positions is a constant - doing electrons in wire first
def visualize_atoms(epositions, evelocities, step, t):
    global gridx, gridy, gridz, bounds, nucleus_positions, electron_speed, electron_velocities  # all these global are really constants
    global visualize_start, visualize_stop, visualize_plane_step

    print("visualize_atoms not working for 2D structures yet")
    return

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
    for x in range(visualize_start, visualize_stop, visualize_plane_step):  # 
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
                # XXXX using abs above and below means that yellow does not always mean to the right.  Hum
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

    plt.close(fig)  # Close the figure to free memory


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

    return(averaged_xdiff.get())    # make Numpy for visualization that runs on CPU



#  Use CuPy to make a histogram of how many electrons are currently in each slice of the wire
def calculate_histogram_positions(epositions):
    global initial_spacing, gridx

    # Get x positions directly from the 2D array (all rows, 0th column for x)
    x_positions = epositions[:, 0]

    # Convert positions to segment indices
    segment_indices = cp.floor(x_positions / initial_spacing).astype(cp.int32)

    # Calculate histogram
    histogram, _ = cp.histogram(segment_indices, bins=cp.arange(-0.5, gridx + 0.5, 1))

    return histogram.get()    # return in Numby not cupy



#  Use CuPy to get average drift velocity of electrons in each slice of the wire
def calculate_drift_velocities(epositions, evelocities):
    global initial_spacing, gridx
    
    # Get x positions and x velocities directly from the 2D arrays
    x_positions = epositions[:, 0]       # x component of positions for each electron
    x_velocities = evelocities[:, 0]     # x component of velocity for each electron

    # Convert positions to segment indices
    segment_indices = cp.floor(x_positions / initial_spacing).astype(cp.int32)

    # Initialize an array to store the sum of velocities in each segment
    velocity_sums = cp.zeros(gridx, dtype=cp.float32)
    
    # Initialize an array to count the number of electrons in each segment
    electron_counts = cp.zeros(gridx, dtype=cp.int32)
    
    # Use bincount to sum velocities and count electrons in each segment
    velocity_sums = cp.bincount(segment_indices, weights=x_velocities, minlength=gridx)
    electron_counts = cp.bincount(segment_indices, minlength=gridx)
    
    # To avoid division by zero, replace zeros in electron_counts with ones (or use np.where to handle zeros)
    electron_counts = cp.where(electron_counts == 0, 1, electron_counts)
    
    # Calculate average velocities
    average_velocities = velocity_sums / electron_counts

    return average_velocities.get()  # Return in NumPy not CuPy






def visualize_wire(histogram, step, t):
    # Plotting
    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    #  Want to plot only betweew wire_start and wire_stop
    # ax.plot(range(wire_start, wire_stop), averaged_xdiff[wire_start:wire_stop], marker='o')
    ax.plot(range(0, len(histogram)), histogram, marker='o')


    ax.set_xlabel('X index')
    ax.set_ylabel('Histogram')
    ax.set_title(f'Step {step} Time: {t:.8e} sec {sim_settings}')
    ax.grid(True)

    # Save the figure
    filename = os.path.join('simulation', f'wire_{step}.png')
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory



def calculate_forces_all():
    global electron_positions, forces, coulombs_constant, electron_charge

    # Number of electrons
    n_electrons = electron_positions.shape[0]

    # Expand electron_positions to calculate pairwise differences (broadcasting)
    delta_r = electron_positions[:, None, :] - electron_positions[None, :, :]
    
    # Calculate distances and handle division by zero
    distances = cp.linalg.norm(delta_r, axis=2)
    cp.fill_diagonal(distances, cp.inf)  # Avoid division by zero for self-interactions
    
    # Calculate forces (Coulomb's Law)
    force_magnitude = coulombs_constant * (electron_charge ** 2) / distances**2
    
    # Normalize force vectors and multiply by magnitude
    unit_vectors = delta_r / distances[:, :, None]  # Add new axis for broadcasting
    normforces = force_magnitude[:, :, None] * unit_vectors
    
    # Sum forces from all other electrons for each electron
    forces = cp.sum(normforces, axis=1)

    # Diagnostic output
    mean_force_magnitude = cp.mean(cp.linalg.norm(forces, axis=1))
    max_force_magnitude = cp.max(cp.linalg.norm(forces, axis=1))
    print("Mean force magnitude:", mean_force_magnitude)
    print("Max force magnitude:", max_force_magnitude)



def update_pv(dt):
    global electron_velocities, electron_positions, bounds, forces, effective_electron_mass

    # Calculate acceleration based on F=ma
    acceleration = forces / effective_electron_mass

    # Update velocities
    electron_velocities += acceleration * dt

    # Update positions using vectors
    electron_positions += electron_velocities * dt

    # Keep positions and velocities within bounds
    for dim in range(3):  # Iterate over x, y, z dimensions
        # Check and apply upper boundary conditions
        over_max = electron_positions[:, dim] > bounds[dim][1]
        electron_positions[over_max, dim] = bounds[dim][1]  # Set to max bound
        electron_velocities[over_max, dim] *= -1  # Reverse velocity

        # Check and apply lower boundary conditions
        below_min = electron_positions[:, dim] < bounds[dim][0]
        electron_positions[below_min, dim] = bounds[dim][0]  # Set to min bound
        electron_velocities[below_min, dim] *= -1  # Reverse velocity

    # Diagnostic output to monitor maximum position and velocity magnitudes
    max_position_magnitude = cp.max(cp.linalg.norm(electron_positions, axis=1))
    max_velocity_magnitude = cp.max(cp.linalg.norm(electron_velocities, axis=1))
    print("Max positions:", max_position_magnitude)
    print("Max velocity:", max_velocity_magnitude)



def main():
    global gridx, gridy, gridz, initial_spacing, num_steps, speedup, forces, electron_positions, electron_velocities, dt

    print("In main")
    checkgpu()
    GPUMem()
    initialize_atoms()
    os.makedirs('simulation', exist_ok=True) # Ensure the simulation directory exists


    # Create a LocalCluster with a custom death timeout and then a Client
    cluster = LocalCluster(n_workers=4, death_timeout='1000s')
    client= Client(cluster)
    futures = []

    print("Doing first visualization")
    copypositions=electron_positions.get()   # get makes Numpy copy so runs on CPU in Dask
    copyvelocities=electron_velocities.get() # get makes Numpy copy so runs on CPU in Dask
    future = client.submit(visualize_atoms, copypositions, copyvelocities, -1, 0.0)
    futures.append(future)
    # main simulation loop
    for step in range(num_steps):
        t = step * dt
        print("In main", step)
        GPUMem()
        if step % WireSteps == 0:
            # WireStatus=calculate_wire(electron_positions)
            # WireStatus=calculate_histogram_positions(electron_positions)
            WireStatus=calculate_drift_velocities(electron_positions, electron_velocities)
            future = client.submit(visualize_wire, WireStatus, step, t)
            futures.append(future)
        if step % DisplaySteps == 0:
            print("Display", step)
            copypositions=electron_positions.get() # get makes Numpy copy so runs on CPU in Dask
            copyvelocities=electron_velocities.get() # get makes Numpy copy so runs on CPU in Dask
            future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t)
            futures.append(future)

        #print("Updating force chunked", step)
        #calculate_forces_chunked()
        #print("Updating force nearby", step)
        #calculate_forces_nearby()
        print("Updating force all", step)
        calculate_forces_all()
        GPUMem()

        print("Updating position and velocity", t)
        update_pv(dt)

        #print("detect and resolve collisions", t)
        #detect_and_resolve_collisions()

        cp.cuda.Stream.null.synchronize()         # free memory on the GPU

    copypositions=electron_positions.get()
    copyvelocities=electron_velocities.get()
    future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t) # If we end at 200 we need last output
    futures.append(future)
    wait(futures)



if __name__ == '__main__':
    main()
