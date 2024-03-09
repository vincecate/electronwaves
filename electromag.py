#!/usr/bin/python3
# Requires:
# pip install numpy
# pip install cupy-cuda115
# pip install matplotlib
# pip install scipy
# pip install dask
# pip install distributed
# pip install pycuda
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
#
# Would be amazing result of this theory if it could correctly show:
#  1- speed of light comes out right from plane wave and electrons charge and mass
#  2- light of certain frequencies can drive electrons off as in experiments
#  3- magnetic field from moving charges - if we can simulate moving charges curving in mag field using instant force beteen electrons
#  4 - ampiers law that two wires next to each othe with current going the same way attract and opposite then repel
#
# At room temperature a ratio of 1 electron in 100,000 copper atoms is really free.
#   Same as saying only 0.001% or 10^-5 of the conduction electrons are free.

#  Update idea Feb 9, 2024
# In classical simulations, introducing a "hard sphere" collision model for very close distances could prevent physical 
#  impossibilities like electron overlap. This involves detecting when electrons are within a certain minimum distance of 
# each other and then calculating their trajectories post-collision based on conservation of momentum and energy, similar to billiard balls colliding.
#
#  In the wire display it may be good to show average movement of each dX in a graph.  Perhaps this will show
#  a much faster wave than what is probably a and "electron density wave".   Feb 12, 2024
#
#  Should make switch to show wire offset/density/velocites   Feb 13, 2024


import cupy as cp
import numpy as np   # numpy as np for CPU and now just for visualization 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.constants import e, epsilon_0, electron_mass, elementary_charge, c    
speed_of_light = c
electron_charge=elementary_charge
coulombs_constant = 8.9875517873681764e9  # Coulomb's constant
#coulombs_constant = 1 / (4 * cp.pi * epsilon_0)  # Coulomb's constant 

import dask
from dask import delayed
from dask.distributed import Client, wait, LocalCluster
import multiprocessing
import os
import sys
import math
import json

# Check if at least one argument is provided (excluding the script name)
if len(sys.argv) > 1:
    simnum = int(sys.argv[1])  # The first argument passed to the script
    print(f"Simulation number provided: {simnum}")
else:
    print("No simulation number provided. Exiting.")
    sys.exit(1)  # Exit the script with an error code

cp.random.seed(999)
# $1 argument gets saved to simnum so can do batch of several simulations from script
# have a set to try to get projection to speed of light in full size plane wave

effective_electron_mass = electron_mass   #  default is the same
electron_thermal_speed = 1.1e6            # meters per second
bounce_distance = 1e-10                   # closer than this and we make electrons bounce off each other

# Making wider wires have deeper pulses so scaling is 3D to give better estimate for real wire extrapolation

# Load the settings from the JSON file
with open('settings.json', 'r') as file:
    settings = json.load(file)

# Access the settings for the given simulation number
if str(simnum) not in settings:  # Convert simnum to string for matching keys
    print(f"No settings found for simulation number {simnum}")
    sys.exit(2)     # Exit with error code

sim_settings = settings[str(simnum)]
gridx = sim_settings.get('gridx', 100 )
gridy = sim_settings.get('gridy', 40 )
gridz = sim_settings.get('gridz', 40 )
speedup = sim_settings.get('speedup', 50)
pulse_width = sim_settings.get('pulse_width', 40 )
num_steps = sim_settings.get('num_steps', 2000 )
forcecalc = sim_settings.get('forcecalc', 1 )               # 1 for CUDA, 2 chunked, 3 nearby, 4 call 
reverse_factor = sim_settings.get('reverse_factor', -0.95)  # when hits side of the wire is reflected - -1=100% and -0.95=95% velocity after reflected
search_type= sim_settings.get('search_type', 1)  # 1 is binary search 2 is "twoshot"
initialize_velocities= sim_settings.get('initialize_velocities', False) # can have electrons initialized to moving if True and not moving if False
use_lorentz= sim_settings.get('use_lorentz', True) # use Lorentz transformation on coulombic force if true 


DisplaySteps = 5000  # every so many simulation steps we call the visualize code
WireSteps = 1        # every so many simulation steps we call the visualize code


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


# bounds format is  ((minx,  maxx) , (miny, maxy), (minz, maxz))
bounds = ((0, gridx*initial_spacing), (0, gridy*initial_spacing), (0, gridz*initial_spacing))

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


# Make string of some settings to put on output graph 
sim_settings = f"simnum {simnum} gridx {gridx} gridy {gridy} gridz {gridz} speedup {speedup} \n Spacing: {initial_spacing:.8e} Pulse Width {pulse_width} ElcSpeed {electron_thermal_speed:.8e} Steps: {num_steps} dt: {dt:.8e} iv:{initialize_velocities} st:{search_type}"

def GPUMem():
    # Get total and free memory in bytes
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()

    print(f"Total GPU Memory: {total_mem / 1e9} GB")
    print(f"Free GPU Memory: {free_mem / 1e9} GB")

GPUMem()



# num_electrons is the total number of electrons
num_electrons = gridx * gridy * gridz
chunk_size = gridx*gridy
# Initialize CuPy/GPU arrays for positions, velocities , and forces as "2D" arrays but really 1D with 3 storage at each index for x,y,z
electron_positions = cp.zeros((num_electrons, 3))
electron_velocities = cp.zeros((num_electrons, 3))
forces = cp.zeros((num_electrons, 3))

past_positions_count = 100
electron_past_positions = cp.zeros((num_electrons, past_positions_count, 3))   # keeping past positions with current at 0, previos 1, etc
electron_past_velocities = cp.zeros((num_electrons, past_positions_count, 3))   # keeping past positions with current at 0, previos 1, etc

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


# When done with initialize_electrons these two arrays should have this shape
# electron_positions = cp.zeros((num_electrons, 3))
# electron_velocities = cp.zeros((num_electrons, 3))
# past_positions_count = 100
# electron_past_positions = cp.zeros((num_electrons, past_positions_count, 3))   # keeping past positions with current at 0, previos 1, etc
def initialize_electrons():
    global initial_radius, electron_velocities, electron_positions, num_electrons, electron_past_positions
    global initial_spacing, initialize_velocities, electron_speed, pulse_width, electron_thermal_speed, gridx, gridy, gridz

    # Calculate the adjusted number of electrons for the pulse and rest regions to match the num_electrons
    # Ensure total electrons do not exceed num_electrons while maintaining higher density in the pulse region
    pulse_volume_ratio = pulse_width / gridx
    adjusted_pulse_electrons = int(num_electrons * pulse_volume_ratio * 2)  # Double density in pulse region
    rest_electrons = num_electrons - adjusted_pulse_electrons

    # Generate random positions for electrons
    x_positions = cp.concatenate([
        cp.random.uniform(0, pulse_width * initial_spacing, adjusted_pulse_electrons),
        cp.random.uniform(pulse_width * initial_spacing, gridx * initial_spacing, rest_electrons)
    ])
    y_positions = cp.random.uniform(0, gridy * initial_spacing, num_electrons)
    z_positions = cp.random.uniform(0, gridz * initial_spacing, num_electrons)

    # Shuffle the x_positions to mix pulse and rest electrons, maintaining overall distribution
    cp.random.shuffle(x_positions)

    # Stack x, y, z positions to form the electron_positions array
    electron_positions = cp.stack((x_positions[:num_electrons], y_positions, z_positions), axis=-1)

    # Initialize velocities
    if initialize_velocities:
        electron_velocities = generate_thermal_velocities(num_electrons, electron_thermal_speed)
    else:
        electron_velocities = cp.zeros((num_electrons, 3))

    
    # Set all past positions to the current positions
    # We use broadcasting to replicate the current positions across the second dimension of electron_past_positions
    # electron_past_positions[:] = electron_positions[:, None, :]

    # Set all past positions to the current positions
    electron_past_positions = cp.tile(electron_positions[:, None, :], (1, past_positions_count, 1))

    electron_past_velocities = cp.tile(electron_velocities[:, None, :], (1, past_positions_count, 1))

    # Explanation:
    # electron_positions[:, None, :] reshapes electron_positions for broadcasting by adding an extra dimension
    # This makes its shape (num_electrons, 1, 3), which is compatible for broadcasting across the past_positions dimension
    # The assignment then replicates the current position across all past positions for each electron



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
def calculate_wire_offset(epositions):
    global gridx, gridy, gridz, nucleus_positions

    # Calculate the difference in the x-direction
    xdiff = epositions[:,:,:,0] - nucleus_positions[:,:,:,0]

    # Calculate the total difference for each x slice
    totalxdiff = cp.sum(xdiff, axis=(1, 2))  # Summing over y and z axes

    # Calculate the average difference for each x slice
    averaged_xdiff = totalxdiff / (gridy * gridz)

    return(averaged_xdiff.get())    # make Numpy for visualization that runs on CPU



#  Use CuPy to make a histogram to get density of how many electrons are currently in each slice of the wire
def calcualte_wire_density(epositions):
    global initial_spacing, gridx

    # Get x positions directly from the 2D array (all rows, 0th column for x)
    x_positions = epositions[:, 0]

    # Convert positions to segment indices
    segment_indices = cp.floor(x_positions / initial_spacing).astype(cp.int32)

    # Calculate histogram
    histogram, _ = cp.histogram(segment_indices, bins=cp.arange(-0.5, gridx + 0.5, 1))

    return histogram.get()    # return in Numby not cupy



#  Use CuPy to get average drift velocity of electrons in each slice of the simulated wire
#  Return as a NumPy array so CPU/Dask output routine can use it
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





# Given an array with values we plot it
#  Can be used for density or average velocity along the wire
def visualize_wire(ylabel, yvalues, step, t):
    # Plotting
    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    #  Want to plot only betweew wire_start and wire_stop
    # ax.plot(range(wire_start, wire_stop), averaged_xdiff[wire_start:wire_stop], marker='o')
    ax.plot(range(0, len(yvalues)), yvalues, marker='o')


    ax.set_xlabel('X index')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Step {step} Time: {t:.8e} sec {sim_settings}')
    ax.grid(True)

    # Save the figure
    filename = os.path.join('simulation', f'wire_{step}.png')
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory




# Because the vector operations were using memory for all pairs of electrons we have
#  broken it down into chunks to limit this memory usage problem.
# After this 'forces' will contain the cumulative forces acting on each electron due to all
def calculate_forces_chunked():
    global electron_positions, electron_velocities, forces, coulombs_constant, electron_charge, num_electrons, chunk_size, speed_of_light
    # Reset forces to zero
    forces.fill(0)

    # Constants
    epsilon = 1e-20  # Small number to avoid division by zero

    # Pre-allocate memory for the largest possible chunk forces, r_squared, r_unit, and time delays
    max_chunk_forces = cp.zeros((chunk_size, num_electrons, 3))
    max_r_squared = cp.zeros((chunk_size, num_electrons))
    max_r_unit = cp.zeros((chunk_size, num_electrons, 3))
    time_delays = cp.zeros((chunk_size, num_electrons))

    for start_idx in range(0, num_electrons, chunk_size):
        end_idx = min(start_idx + chunk_size, num_electrons)
        chunk_positions = electron_positions[start_idx:end_idx]
        chunk_velocities = electron_velocities[start_idx:end_idx]

        current_chunk_size = end_idx - start_idx

        # Calculate the vector differences between positions in the chunk and all positions
        r_ij = chunk_positions[:, None, :] - electron_positions[None, :, :]
        v_ij = chunk_velocities[:, None, :] - electron_velocities[None, :, :]

        # Compute squared distances
        r_squared = cp.sum(r_ij ** 2, axis=2) + epsilon
        max_r_squared[:current_chunk_size] = r_squared

        # Estimate time delays based on separation distance and speed of light
        time_delays[:current_chunk_size] = cp.sqrt(r_squared) / speed_of_light

        # Estimate the retarded positions based on time delays and relative velocities
        retarded_positions = electron_positions[None, :, :] - v_ij * time_delays[:current_chunk_size, :, None]

        # Recalculate vector differences using retarded positions
        r_ij_retarded = chunk_positions[:, None, :] - retarded_positions
        r_squared_retarded = cp.sum(r_ij_retarded ** 2, axis=2) + epsilon

        # Calculate unit vector for retarded positions
        r_unit_retarded = r_ij_retarded / cp.sqrt(r_squared_retarded)[:, :, None]

        # Calculate relative velocity direction (towards or away)
        relative_velocity_direction = cp.sum(v_ij * r_unit_retarded, axis=2)

        # Calculate adjustment factor based on relative speed and direction  - approaching velocity is negative so need to negate
        adjustment_factor = 1 - (relative_velocity_direction / speed_of_light)

        # Adjust force magnitudes based on direction
        force_magnitudes = adjustment_factor * coulombs_constant * (electron_charge ** 2) / r_squared_retarded 

        # Ensure force_magnitudes is broadcastable to the shape of r_unit_retarded
        # by adding a new axis to force_magnitudes, making it (2100, 63000, 1)
        # This allows for element-wise multiplication with r_unit_retarded (2100, 63000, 3)
        force_magnitudes_expanded = force_magnitudes[:, :, None]  # Add an extra dimension

        # Now perform the multiplication
        max_chunk_forces[:current_chunk_size] = force_magnitudes_expanded * r_unit_retarded

        # Sum forces for the current chunk and add to the total forces
        forces[start_idx:end_idx] += cp.sum(max_chunk_forces[:current_chunk_size], axis=1)



#  These vector operations use storage scaling with the number of pairs of electorns, which is huge
#  It seems there should be a way to just sum of the forces from all other electrons without 
#   needing memory for all pairs of electrons but I don't have that figured out yet.
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



# CUDA kernel
kernel_code = '''
#include <math_functions.h>

const double speed_of_light = 299792458; // Speed of light in meters per second

__device__ double distance3(double3 pos1, double3 pos2) {
        // Calculate the Euclidean distance between the current and past positions
        double dx = pos1.x - pos2.x;
        double dy = pos1.y - pos2.y;
        double dz = pos1.z - pos2.z;
        double distance = sqrt(dx * dx + dy * dy + dz * dz);
        return(distance);
}

// Find how far back in history is best match of distance and delay from speed of light
// Electrons usually don't move much per time slice so this probably gets very close very fast
__device__ int find_best_delay_position_2shot(const double3 current_position, const double3* historical_positions, int history_slices, double dt) {

    int guess = 0;              // In historical_positons 0 has the most recent positon for that electron
    int numshots = 2;
    double distance;
    double ideal_travel_time;
    for (int i=0; i<numshots; i++) {
        distance = distance3(current_position, historical_positions[guess]); // see distance at time guess 
        ideal_travel_time = distance / speed_of_light;
        guess = ideal_travel_time / dt;                                      // and see how many time slices back in time that should be
        guess = max(guess, 0);                                               // bounds checking
        guess = min(guess, history_slices - 1);
    }

    return(guess);
}

// Find how far back in history is best match of distance and delay from speed of light
__device__ int find_best_delay_position_binary(const double3 current_position, const double3* historical_positions, int history_slices, double dt) {
    int left = 0;
    int right = history_slices - 1;

    while (left < right ) {
        int mid = (left + right) / 2;
        if (mid == left || mid == right)
            return mid;                   // we are at the end

        double distance = distance3(current_position, historical_positions[mid]);

        // Calculate the time it takes for light to travel this distance
        double ideal_travel_time = distance / speed_of_light;

        // Calculate how far back in time this historical position is
        double simulation_time = mid * dt;

        if (fabs(ideal_travel_time - simulation_time) < 0.5*dt) {
            // If the difference is less than one half time step, consider it a match
            return mid;
        } else if (simulation_time < ideal_travel_time) {
            // If sim time is not enough for this distance then go further back in time
            left = mid + 1;
        } else {
            // If sim time is too far back for this distance then look closer to the present 
            right = mid - 1;
        }
    }

    // We may never have difference less than dt so just return right in that case
    // printf("left %d  right %d \\n", left, right);
    return right;
}



extern "C" __global__ void calculate_forces(const double3* electron_positions, const double3* electron_velocities,
                                const double3* electron_past_positions,
                                const double3* electron_past_velocities,
                                const int past_positions_count,
                                double3* forces,
                                int num_electrons,
                                double coulombs_constant,
                                double electron_charge,
                                double dt,
                                int search_type,
                                bool use_lorentz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double speed_of_light = 299792458.0; // Speed of light in meters per second

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[CUDA-Cupy] num_electrons %d\\n", num_electrons );  // print once per kernel launch

    if (i < num_electrons) {
        double3 force = make_double3(0.0, 0.0, 0.0);
        double3 current_position = electron_positions[i];
        double3 current_velocity = electron_velocities[i];

        for (int j = 0; j < num_electrons; j++) {
            if (i != j) {
                int best_delay_index;
                if (search_type == 1){
                    best_delay_index = find_best_delay_position_binary(current_position, &electron_past_positions[j * past_positions_count], past_positions_count, dt);
                } else {
                    best_delay_index = find_best_delay_position_2shot(current_position, &electron_past_positions[j * past_positions_count], past_positions_count, dt);
                }
                if (threadIdx.x == 0 && blockIdx.x == 0 && j == 8)     
                    printf("best_delay_index %d\\n", best_delay_index);   // one sample to see it changes

                double3 delayed_position = electron_past_positions[j * past_positions_count + best_delay_index];
                double3 past_velocity = electron_past_velocities[j * past_positions_count + best_delay_index];

                double3 r = make_double3(
                    current_position.x - delayed_position.x,
                    current_position.y - delayed_position.y,
                    current_position.z - delayed_position.z);

                double dist_sq = r.x * r.x + r.y * r.y + r.z * r.z;
                dist_sq = max(dist_sq, 1.0e-30); // avoid divide by zero

                double len_inv = rsqrt(dist_sq); // reciprocal square root
                double3 normalized_r = make_double3(r.x * len_inv, r.y * len_inv, r.z * len_inv);

                double3 relative_velocity = make_double3(
                    current_velocity.x - past_velocity.x,
                    current_velocity.y - past_velocity.y,
                    current_velocity.z - past_velocity.z);


                double relative_velocity_magnitude = sqrt(relative_velocity.x * relative_velocity.x +
                                           relative_velocity.y * relative_velocity.y +
                                           relative_velocity.z * relative_velocity.z + 1.0e-50); // Added epsilon to avoid division by zero
                relative_velocity_magnitude = min(relative_velocity_magnitude, 0.99*speed_of_light);

                double coulomb;
                if (use_lorentz) {
                    // Calculate the Lorentz factor (gamma)
                    double gamma = 1.0 / sqrt(1.0 - (relative_velocity_magnitude * relative_velocity_magnitude) / (speed_of_light * speed_of_light));
                    if (threadIdx.x == 0 && blockIdx.x == 0 && j == 8)     
                        printf("gamma  =%.15lf\\n", gamma);
                    coulomb = gamma * coulombs_constant * electron_charge * electron_charge / dist_sq;
                } else {
                    double dot_product = relative_velocity.x * normalized_r.x +
                                         relative_velocity.y * normalized_r.y +
                                         relative_velocity.z * normalized_r.z;

                    // Ensure dot_product is scaled properly relative to the magnitudes and speed of light
                    // Adjustment_factor should be greater than 1 if coming together and less than 1 if moving away 
                    double adjustment_factor = 1.0; // Default to no adjustment
                
                    // Compute the magnitude of the relative velocity scaled by the speed of light
                    double speed_ratio = relative_velocity_magnitude / speed_of_light;
                    if (threadIdx.x == 0 && blockIdx.x == 0 && j == 8)     
                        printf("speed_ratio =%.15lf\\n", speed_ratio);

                    double speed_ratio_bounded = fmin(0.5, fabs(speed_ratio));  // so positive and bounded between 0 and 0.5

                    // Use the dot product to determine if the electrons are moving towards or away from each other
                    bool movingTowardsEachOther = dot_product < 0;

                    // Adjust the adjustment_factor based on the direction of movement
                    // Increase when moving towards each other, decrease when moving away
                    if (movingTowardsEachOther) {
                        adjustment_factor = 1.0 + speed_ratio_bounded; // increases the force if moving towards each other
                    } else {
                        adjustment_factor = 1.0 - speed_ratio_bounded; // Reduce the force if moving away from each other
                    }


                    // adjustment_factor should now be between 0.5 and 1.5 
                    if (threadIdx.x == 0 && blockIdx.x == 0 && j == 8)     
                        printf("adjustment_factor=%.15lf\\n", adjustment_factor);

                    coulomb = adjustment_factor * coulombs_constant * electron_charge * electron_charge / dist_sq; // dist_sq is non zero
                }

                force.x += coulomb * normalized_r.x;
                force.y += coulomb * normalized_r.y;
                force.z += coulomb * normalized_r.z;
            }
        }
        // Debug: Print the calculated force for the last few electrons
        if(i > num_electrons-10) {
            printf("Electron %d Force: x=%e, y=%e, z=%e\\n", i, force.x, force.y, force.z);
         }

        // Apply the calculated forces   forces[i]=force     - probably good enough
        atomicAdd(&forces[i].x, force.x);
        atomicAdd(&forces[i].y, force.y);
        atomicAdd(&forces[i].z, force.z);
    }
}


'''

# Load CUDA source and get kernel function
module = cp.RawModule(code=kernel_code)
calculate_forces = module.get_function('calculate_forces')

# Kernel launch configuration
threadsperblock = 512
blockspergrid = math.ceil(num_electrons/threadsperblock)


# Note for reference the CuPy globals on the GPU memory are:
#   electron_positions = cp.zeros((num_electrons, 3))
#   electron_velocities = cp.zeros((num_electrons, 3))
#   forces = cp.zeros((num_electrons, 3))
#
def calculate_forces_cuda():
    global electron_positions, electron_velocities, electron_past_positions, electron_past_velocities, past_positions_count, forces, num_electrons, coulombs_constant, electron_charge, dt, search_type, use_lorentz
    # Launch kernel with the corrected arguments passing
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    # Launch kernel with the corrected arguments passing
    print("starting calculate_forces_cuda")
    forces.fill(0)
    try:
        start_gpu.record()
        calculate_forces(grid=(blockspergrid, 1, 1),
                     block=(threadsperblock, 1, 1),
                     args=(electron_positions, electron_velocities, electron_past_positions, electron_past_velocities, past_positions_count, forces, num_electrons, coulombs_constant, electron_charge,dt, search_type, use_lorentz))
        cp.cuda.Device().synchronize()    #  Let this finish before we do anything else

        end_gpu.record()
        end_gpu.synchronize()
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        print("Calculate_forces duration:", t_gpu)

    except cp.cuda.CUDARuntimeError as e:
        print(f"CUDA Error: {e}")
    print("ending calculate_forces_cuda")

def print_forces_sum():
    global forces, num_electrons

    # Sum forces along the first axis to get the total force in each direction
    total_forces = cp.sum(forces, axis=0)

    # total_forces now contains the sum of x, y, and z components of the forces
    # Calculate the magnitude of the total force vector
    total_force_magnitude = cp.linalg.norm(total_forces)
    print("total_force_magnitude = ", total_force_magnitude)


def update_pv(dt):
    global electron_velocities, electron_positions, bounds, forces, effective_electron_mass, electron_past_positions

    # Calculate acceleration based on F=ma
    acceleration = forces / effective_electron_mass

    # Update velocities
    electron_velocities += acceleration * dt

    # Update positions using vectors
    electron_positions += electron_velocities * dt

    # Keep positions and velocities within bounds
    for dim in range(3):  # Iterate over x, y, z dimensions
        # Check and apply upper boundary conditions
        over_max = electron_positions[:, dim] > bounds[dim][1]   # 1 holds max
        electron_positions[over_max, dim] = bounds[dim][1]  # Set to max bound
        electron_velocities[over_max, dim] *= reverse_factor  # Reverse velocity

        # Check and apply lower boundary conditions
        below_min = electron_positions[:, dim] < bounds[dim][0]  # 0 holds min
        electron_positions[below_min, dim] = bounds[dim][0]  # Set to min bound
        electron_velocities[below_min, dim] *= reverse_factor  # Reverse velocity

    
    # Step 1: Shift all past positions to the right by one position
    # We copy from the end towards the beginning to avoid overwriting data that we still need to copy
    for i in range(past_positions_count - 1, 0, -1):
        electron_past_positions[:, i, :] = electron_past_positions[:, i - 1, :]
        electron_past_velocities[:, i, :] = electron_past_velocities[:, i - 1, :]

    # Step 2: Copy the current electron_positions into the first spot of electron_past_positions and past_velocities
        electron_past_positions[:, 0, :] = electron_positions
        electron_past_velocities[:, 0, :] = electron_velocities


    # Diagnostic output to monitor maximum position and velocity magnitudes
    max_position_magnitude = cp.max(cp.linalg.norm(electron_positions, axis=1))
    max_velocity_magnitude = cp.max(cp.linalg.norm(electron_velocities, axis=1))
    print("Max positions:", max_position_magnitude)
    print("Max velocity:", max_velocity_magnitude)



def main():
    global gridx, gridy, gridz, initial_spacing, num_steps, speedup, forces, electron_positions, electron_velocities, dt

    print("In main")
    if (pulse_width > gridx/2):
        print("pulse_width has to be less than half of gridx")
        exit(-1)

    checkgpu()
    GPUMem()
    initialize_electrons()
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
            # WireStatus=calculate_wire_offset(electron_positions)
            # future = client.submit(visualize_wire, "Offset",  WireStatus, step, t)
            WireStatus=calcualte_wire_density(electron_positions)
            future = client.submit(visualize_wire, "Density", WireStatus, step, t)
            #WireStatus=calculate_drift_velocities(electron_positions, electron_velocities)
            #future = client.submit(visualize_wire, "Velocity", WireStatus, step, t)
            futures.append(future)
        if step % DisplaySteps == 0:
            print("Display", step)
            copypositions=electron_positions.get() # get makes Numpy copy so runs on CPU in Dask
            copyvelocities=electron_velocities.get() # get makes Numpy copy so runs on CPU in Dask
            future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t)
            futures.append(future)

        print("main before calling forces")
        print_forces_sum()
        if (forcecalc == 1):
            print("calculate_forces_cuda", step)
            calculate_forces_cuda()
        if (forcecalc == 2):
            print("Updating force chunked", step)
            calculate_forces_chunked()
        if (forcecalc == 3): 
            print("Updating force nearby", step)
            calculate_forces_nearby()
        if (forcecalc == 4): 
            print("Updating force all", step)
            calculate_forces_all()
        print("main after calling forces")
        print_forces_sum()
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
