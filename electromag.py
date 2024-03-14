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
#  We currently 3/9/24 only simulate free electrons but the "displacement current" from bound electrons might be important
#  We can simulate electrons bound to atoms as electrons on a spring to a point.  So this would not be hard.
#  Can have an array of flags telling us which electrons are bound
#  And another array of atom_positions showing where the spring is connected to
#  This was maxwell's model and so almost certain to work for propagating waves
#  
# Gemini:
#     Electromagnetic waves like light interact much more strongly with bound electrons than typical electrical signals 
#     in a wire. This is the basis for how materials reflect, refract, and absorb light.
#  So to do light we expect to need to simulate bound electrons but we can easily do this I think.
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
import time
from datetime import datetime, timedelta

# $1 argument gets saved to simnum so can do batch of several simulations from script
# Check if at least one argument is provided (excluding the script name)
if len(sys.argv) > 1:
    simnum = int(sys.argv[1])  # The first argument passed to the script
    print(f"Simulation number provided: {simnum}")
else:
    print("No simulation number provided. Exiting.")
    sys.exit(1)  # Exit the script with an error code

# Setting a random seed makes same run do the same thing
# cp.random.seed(999)



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
pulse_width = sim_settings.get('pulse_width', 0 )
num_steps = sim_settings.get('num_steps', 2000 )
forcecalc = sim_settings.get('forcecalc', 1 )               # 1 for CUDA, 2 chunked, 3 nearby, 4 call 
reverse_factor = sim_settings.get('reverse_factor', -0.95)  # when hits side of the wire is reflected - -1=100% and -0.95=95% velocity after reflected
search_type= sim_settings.get('search_type', 2)  #  0 is no history, 1 is binary search of history, 2 is "twoshot" in history
initialize_velocities= sim_settings.get('initialize_velocities', False) # can have electrons initialized to moving if True and not moving if False
use_lorentz= sim_settings.get('use_lorentz', False) # use Lorentz transformation on coulombic force if true 
filename_load = sim_settings.get('filename_load', "none") # Can save or load electron positions and velocities - need right num_electrons
filename_save = sim_settings.get('filename_save', "simulation.data") # Can save or load electron positions and velocities - need right num_electrons
pulse_density = sim_settings.get('pulse_density', 1) # Can save or load electron positions and velocities - need right num_electrons
max_velocity = sim_settings.get('max_velocity', 0.95*speed_of_light) # Speed limit for electrons 
boltz_temp = sim_settings.get('boltz_temp', 300.0)        # Boltzman temperature for random velocities 
wire_steps = sim_settings.get('wire_steps', 1)            # How many steps between wire plot outputs 
display_steps = sim_settings.get('display_steps', 8000)   # every so many simulation steps we call the visualize code
past_positions_count = sim_settings.get('past_positions_count', 100)   # how many past positions history we keep for each electron
initialize_wave = sim_settings.get('initialize_wave', True)   # Try to initialize in a wave pattern so not in rush to move 
pulse_velocity = sim_settings.get('pulse_velocity', 0)   # Have electrons in pulse area moving
pulse_offset = sim_settings.get('pulse_offset', 0)   # X value offset for pulse 
force_velocity_adjust = sim_settings.get('force_velocity_adjust', True)   # X value offset for pulse 
velocity_cap = sim_settings.get('velocity_cap', 3e7)         # For the cappedvelocity output we ignore faster than this
collision_distance = sim_settings.get('collision_distance', 1e-10)  # Less than this simulate a collision
collision_on = sim_settings.get('collision_on', True)  # Simulate collisions 
collision_max = sim_settings.get('collision_max', 1000) # Maximum number of collisions per time slice

effective_electron_mass = electron_mass   #  default is the same
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
bounds = ((0, (gridx-1)*initial_spacing), (0, (gridy-1)*initial_spacing), (0, (gridz-1)*initial_spacing))

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
sim_settings = f"simnum {simnum} gridx {gridx} gridy {gridy} gridz {gridz} speedup {speedup} lorentz {use_lorentz}  \n Spacing: {initial_spacing:.8e} Pulse Width {pulse_width} Steps: {num_steps} dt: {dt:.8e} iv:{initialize_velocities} st:{search_type}"

def GPUMem():
    # Get total and free memory in bytes
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()

    print(f"Total GPU Memory: {total_mem / 1e9} GB")
    print(f"Free GPU Memory: {free_mem / 1e9} GB")

GPUMem()



# Allocate memory for collision pairs (2 integers per collision)
collision_pairs = cp.zeros((collision_max, 2), dtype=cp.int32)

# Allocate memory for the collision count (a single integer)
collision_count = cp.zeros(1, dtype=cp.int32)



# num_electrons is the total number of electrons
num_electrons = gridx * gridy * gridz
chunk_size = gridx*gridy
# Initialize CuPy/GPU arrays for positions, velocities , and forces as "2D" arrays but really 1D with 3 storage at each index for x,y,z
electron_positions = cp.zeros((num_electrons, 3))
electron_velocities = cp.zeros((num_electrons, 3))
forces = cp.zeros((num_electrons, 3))
electron_is_bound = cp.zeros(num_electrons, dtype=bool)   # if True then electron is bound to an atom, if flase then free
electron_atom_center = cp.zeros((num_electrons, 3))       # atom position that spring for bound electron is attached to

electron_past_positions = cp.zeros((num_electrons, past_positions_count, 3))   # keeping past positions with current at 0, previos 1, etc
electron_past_velocities = cp.zeros((num_electrons, past_positions_count, 3))   # keeping past positions with current at 0, previos 1, etc



def save_arrays():
    global filename_save, electron_positions, electron_velocities
    """
    Saves the electron positions and velocities to a file.

    Parameters:
    - filename_save: The filename where the arrays will be saved.
    - electron_positions: CuPy array of electron positions.
    - electron_velocities: CuPy array of electron velocities.
    """
    # Convert CuPy arrays to NumPy arrays for saving.
    electron_positions_np = cp.asnumpy(electron_positions)
    electron_velocities_np = cp.asnumpy(electron_velocities)
    
    # Use numpy's savez to save multiple arrays to a file in compressed format.
    np.savez_compressed(filename_save, positions=electron_positions_np, velocities=electron_velocities_np)

    print(f"save_arrays saved to {filename_save}")

   
def load_arrays():
    global filename_load, electron_positions, electron_velocities, num_electrons
    """
    Loads the electron positions and velocities from a file into NumPy arrays,
    validates their sizes, and if validation passes, copies them to CuPy arrays.

    Parameters:
    - state_filename: The filename from where the arrays will be loaded.
    - expected_num_electrons: The expected number of electrons (items) in the arrays.

    Returns:
    - electron_positions_cp: CuPy array of electron positions if validation passes, else None.
    - electron_velocities_cp: CuPy array of electron velocities if validation passes, else None.
    - validation_passed: Boolean indicating whether the validation passed.
    """
    if (filename_load == "none"):
        print("load_arrays NOT loading from file")
        return
    print(f"load_arrays loading from file {filename_load}")

    data = np.load(filename_load)
    electron_positions_np = data['positions']
    electron_velocities_np = data['velocities']

    # Validate the size of the loaded arrays
    positions_correct = electron_positions_np.shape[0] == num_electrons
    velocities_correct = electron_velocities_np.shape[0] == num_electrons

    if (positions_correct and velocities_correct):                 # If validation passes, convert NumPy arrays to CuPy arrays
        electron_positions = cp.asarray(electron_positions_np)
        electron_velocities = cp.asarray(electron_velocities_np)
    else:
        print(f"Aborting:  Loading from file {filename_load} did not get right number of electrons ")
        sys.exit(-1)                                                   # user settings are wrong abort
    print("load_arrays successful")




# Note  - CUDA kerel has made a list of collision_pairs that is collision_count long
# electron_positions = cp.zeros((num_electrons, 3))
# electron_velocities = cp.zeros((num_electrons, 3))
def resolve_collisions():
    global electron_positions, electron_velocities, collision_count, collision_pairs

    # Read the number of collisions detected
    num_collisions = collision_count.item()  # Convert to a Python scalar

    # Read the collision pairs, and slice based on the actual number of collisions
    collision_pairs_np = collision_pairs[:num_collisions].get()

    print(f"Number of collisions: {num_collisions}")
    print("Collision pairs (electron indexes):")
    print(collision_pairs_np)

    # Ensure each pair is sorted
    sorted_pairs = np.sort(collision_pairs_np, axis=1)

    # Use np.unique to remove duplicates. Since np.unique works on 1D arrays,
    # we view the 2D array as a structured array to treat each row as an element.
    dtype = [('first', sorted_pairs.dtype), ('second', sorted_pairs.dtype)]
    unique_pairs = np.unique(sorted_pairs.view(dtype))

    # Convert back to a 2D array
    unique_pairs = unique_pairs.view(sorted_pairs.dtype).reshape(-1, 2)

    print("Unique collision pairs:")
    print(unique_pairs)

    for i in range(len(unique_pairs)):
        e1, e2 = unique_pairs[i]
        collision_vector = electron_positions[e1] - electron_positions[e2]
        collision_vector /= cp.linalg.norm(collision_vector)

        v1 = electron_velocities[e1]
        v2 = electron_velocities[e2]

        # Calculate the projections of the velocities onto the collision vector
        v1_proj = cp.dot(v1, collision_vector)
        v2_proj = cp.dot(v2, collision_vector)

        # Swap the velocity components along the collision vector (elastic collision)
        v1_new = v1 - v1_proj * collision_vector + v2_proj * collision_vector
        v2_new = v2 - v2_proj * collision_vector + v1_proj * collision_vector

        electron_velocities[e1] = v1_new
        electron_velocities[e2] = v2_new

    # Zero out collision count for the next iteration
    collision_count.fill(0)



def generate_thermal_velocities():
    global num_electrons, boltz_temp
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
    sigma = cp.sqrt(kb * boltz_temp / electron_mass)

    # Generate random speeds from a Maxwell-Boltzmann distribution
    # Use the fact that the Maxwell-Boltzmann distribution for speeds in one dimension
    # is a normal distribution with mean 0 and standard deviation sigma
    vx = cp.random.normal(loc=0, scale=sigma, size=num_electrons)
    vy = cp.random.normal(loc=0, scale=sigma, size=num_electrons)
    vz = cp.random.normal(loc=0, scale=sigma, size=num_electrons)

    # Combine the velocity components into a single 2D array
    velocities = cp.stack((vx, vy, vz), axis=-1)

    return velocities


def initialize_electrons_sine_wave():
    global initial_radius, electron_velocities, electron_positions, num_electrons, electron_past_positions
    global initial_spacing, initialize_velocities, electron_speed, pulse_width, gridx, gridy, gridz, past_positions_count

    print("initialize_electrons_sine_wave")

    # Generate positions for electrons with more density towards the ends
    # This uses a sine wave function to skew the distribution towards 0 and gridx
    n=1.0
    theta = cp.linspace(0, cp.pi, num_electrons)
    x_positions_skewed = (gridx-1) * ((1 - cp.cos(theta))**n / (2**n)) * initial_spacing   # XXXX not sure gridx-1 might be better gridx
    y_positions = cp.random.uniform(0, gridy * initial_spacing, num_electrons)
    z_positions = cp.random.uniform(0, gridz * initial_spacing, num_electrons)

    # Stack x, y, z positions to form the electron_positions array
    electron_positions = cp.stack((x_positions_skewed, y_positions, z_positions), axis=-1)


    # Initialize velocities if needed
    if initialize_velocities:
        electron_velocities = generate_thermal_velocities()
    else:
        electron_velocities = cp.zeros((num_electrons, 3))

    pulse_electrons = pulse_width*gridy*gridz       # number of electrons in pulse volume
    pulse_start = pulse_offset*initial_spacing
    pulse_end = pulse_start + (pulse_width*initial_spacing)
    print(f"pulse_electrons {pulse_electrons}")     # 
    print(f"pulse_start {pulse_start}")     # 
    print(f"pulse_end {pulse_end}")     # 
    if (pulse_velocity > 0):
        xp = electron_positions[:, 0]                   # Extract the x positions of all electrons
        mask = (xp > pulse_start) & (xp < pulse_end)    # Create a boolean mask for the condition
        electron_velocities[mask, 0] = pulse_velocity   # Apply the mask to set the x velocities

    
    # Initialize past positions array
    electron_past_positions = cp.tile(electron_positions[:, None, :], (1, past_positions_count, 1))

    # Additional steps to set up velocities and past positions as needed...


# When done with initialize_electrons these two arrays should have this shape
# electron_positions = cp.zeros((num_electrons, 3))
# electron_velocities = cp.zeros((num_electrons, 3))
# past_positions_count = 100
# electron_past_positions = cp.zeros((num_electrons, past_positions_count, 3))   # keeping past positions with current at 0, previos 1, etc
def initialize_electrons():
    global initial_radius, electron_velocities, electron_positions, num_electrons, electron_past_positions, pulse_density
    global initial_spacing, initialize_velocities, electron_speed, pulse_width, gridx, gridy, gridz

    # Calculate the adjusted number of electrons for the pulse and rest regions to match the num_electrons
    # Ensure total electrons do not exceed num_electrons while maintaining higher density in the pulse region
    pulse_volume_ratio = pulse_width / gridx
    adjusted_pulse_electrons = int(num_electrons * pulse_volume_ratio * pulse_density)  # Double density in pulse region
    rest_electrons = num_electrons - adjusted_pulse_electrons

    # Generate random positions for electrons
    x_positions = cp.concatenate([
        cp.random.uniform(0, pulse_width * initial_spacing, adjusted_pulse_electrons),
        cp.random.uniform(pulse_width * initial_spacing, gridx * initial_spacing, rest_electrons)
    ])
    y_positions = cp.random.uniform(0, gridy * initial_spacing, num_electrons)
    z_positions = cp.random.uniform(0, gridz * initial_spacing, num_electrons)

    # Shuffle the x_positions to mix pulse and rest electrons, maintaining overall distribution
    # cp.random.shuffle(x_positions)    # XXX seems to defeat the idea of the pulse

    # Stack x, y, z positions to form the electron_positions array
    electron_positions = cp.stack((x_positions[:num_electrons], y_positions, z_positions), axis=-1)

    # Initialize velocities
    if initialize_velocities:
        electron_velocities = generate_thermal_velocities()
    else:
        electron_velocities = cp.zeros((num_electrons, 3))

    load_arrays()   #   we are not saving past_positions to file so far - if no file it  will not change arrays
    
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





#  returns  (density, velocity, amps, speed, cappedvelocity)    - for plotting to files
def calculate_plots():
    global electron_positions, electron_velocities, gridx, gridy, gridz, initial_spacing, electron_charge, velocity_cap

    # Get x positions and velocities from the 2D arrays
    x_positions = electron_positions[:, 0]
    x_velocities = electron_velocities[:, 0]

    # Convert positions to segment indices based on the total length of the wire and number of slices
    segment_indices = cp.floor(x_positions / initial_spacing).astype(cp.int32)
    # Ensure segment indices are within bounds
    segment_indices = cp.clip(segment_indices, 0, gridx - 1)

    # Calculate speeds (magnitude of velocity) for each electron
    speeds = cp.sqrt(cp.sum(electron_velocities**2, axis=1))

    # Use bincount to sum xvelocities, sum speeds, and count electrons in each segment
    xvelocity_sums = cp.bincount(segment_indices, weights=x_velocities, minlength=gridx)
    speed_sums = cp.bincount(segment_indices, weights=speeds, minlength=gridx)
    electron_counts = cp.bincount(segment_indices, minlength=gridx)
    # Avoid division by zero
    electron_counts_nonzero = cp.where(electron_counts == 0, 1, electron_counts)
    
    # Calculate average velocities and speeds
    average_xvelocities = xvelocity_sums / electron_counts_nonzero
    average_speeds = speed_sums / electron_counts_nonzero

    # Create a mask for velocities <= 3e6
    capped_velocity_mask = cp.abs(x_velocities) <= velocity_cap
    # Apply mask to segment indices and velocities
    capped_segment_indices = segment_indices[capped_velocity_mask]
    capped_x_velocities = x_velocities[capped_velocity_mask]
    
    # Calculate sums of capped velocities
    capped_xvelocity_sums = cp.bincount(capped_segment_indices, weights=capped_x_velocities, minlength=gridx)
    capped_electron_counts = cp.bincount(capped_segment_indices, minlength=gridx)
    capped_electron_counts_nonzero = cp.where(capped_electron_counts == 0, 1, capped_electron_counts)
    
    # Calculate average velocities with cap
    capped_average_xvelocities = capped_xvelocity_sums / capped_electron_counts_nonzero

    # Calculate wire slice volume and electron density
    wire_slice_volume = initial_spacing * (gridy * initial_spacing) * (gridz * initial_spacing)
    electron_density = electron_counts / wire_slice_volume

    # Calculate current in each segment
    amps = electron_density * average_xvelocities * electron_charge

    return electron_counts.get(), average_xvelocities.get(), amps.get(), average_speeds.get(), capped_average_xvelocities.get()






# Given an array with values we plot it
#  Can be used for offset, density, velocity, current  along the wire
#  use ylabel.lower() for directory name as well
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
    directory=ylabel.lower()
    filename = os.path.join(directory, f'wire_{step}.png')
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory


def visualize_all_plots(step, t, plots):
    """
    Function to handle visualization of all plots.

    :param step: The current simulation step.
    :param t: The current simulation time.
    :param plots: A list of tuples, each containing the label and the values for a plot.
    """
    for ylabel, yvalues in plots:
        visualize_wire(ylabel, yvalues, step, t)



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
// We look at the current position, see how long to go that far at speed of light to get approximate time back in simulation
//  then go to that far in past and get position at that point in simulation, then make new estimate for distance and time
__device__ int find_best_delay_position_2shot(const double3 current_position, const double3* historical_positions, int history_slices, double dt) {

    int approximation = 0;              // In historical_positons 0 has the most recent positon for that electron
    int numshots = 2;
    double distance;
    double ideal_travel_time;
    for (int i=0; i<numshots; i++) {
        distance = distance3(current_position, historical_positions[approximation]); // see distance at time approximation 
        ideal_travel_time = distance / speed_of_light;
        approximation = ideal_travel_time / dt;                                      // and see how many time slices back in time that should be
        approximation = max(approximation, 0);                                               // bounds checking
        approximation = min(approximation, history_slices - 1);
    }

    return(approximation);
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
                                bool use_lorentz,
                                bool force_velocity_adjust,
                                int collision_max,
                                int* collision_count,
                                int* collision_pairs,
                                double collision_distance) {
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
                switch (search_type) {
                    case 0:
                        best_delay_index =0;
                        break;
                    case 1:
                        best_delay_index = find_best_delay_position_binary(current_position, &electron_past_positions[j * past_positions_count], past_positions_count, dt);
                        break;
                    case 2:
                        best_delay_index = find_best_delay_position_2shot(current_position, &electron_past_positions[j * past_positions_count], past_positions_count, dt);
                        break;
                    default:
                        printf("ERROR search_type is not valid %d\\n", search_type);
                        best_delay_index =0;
                        break;

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
                if (dist_sq < collision_distance * collision_distance) {
                    int collision_id = atomicAdd(collision_count, 1);
                    if (collision_id < collision_max) {
                        collision_pairs[2 * collision_id] = i;           // record this electron
                        collision_pairs[2 * collision_id + 1] = j;       // and one we collide with
                    }
                }
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
                double dot_product = relative_velocity.x * normalized_r.x +
                                     relative_velocity.y * normalized_r.y +
                                     relative_velocity.z * normalized_r.z;

                // Ensure dot_product is scaled properly relative to the magnitudes and speed of light
                // Adjustment_factor should be greater than 1 if coming together and less than 1 if moving away 
                double adjustment_factor = 1.0; // Default to no adjustment
               
                if (force_velocity_adjust){
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
                        if (use_lorentz) {                   // Calculate the Lorentz factor (gamma)
                            adjustment_factor = 1.0 / sqrt(1.0 - (relative_velocity_magnitude * relative_velocity_magnitude) / (speed_of_light * speed_of_light));
                        } else{
                            adjustment_factor = 1.0 + speed_ratio_bounded; // increases the force if moving towards each other
                        }
                    } else {
                        adjustment_factor = 1.0 - speed_ratio_bounded; // Reduce the force if moving away from each other
                    }
                }


                // adjustment_factor should now be between 0.5 or greater  
                if (threadIdx.x == 0 && blockIdx.x == 0 && j == 8)     
                    printf("adjustment_factor=%.15lf\\n", adjustment_factor);

                coulomb = adjustment_factor * coulombs_constant * electron_charge * electron_charge / dist_sq; // dist_sq is non zero

                force.x += coulomb * normalized_r.x;
                force.y += coulomb * normalized_r.y;
                force.z += coulomb * normalized_r.z;
            } // if less than num_electrons
        }    // for loop
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
    global electron_positions, electron_velocities, electron_past_positions, electron_past_velocities, past_positions_count, forces, num_electrons, coulombs_constant, electron_charge, dt, search_type, use_lorentz, force_velocity_adjust, collision_max, collision_count, collision_pairs, collision_distance
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
                     args=(electron_positions, electron_velocities, electron_past_positions, electron_past_velocities, past_positions_count, forces, num_electrons, coulombs_constant, electron_charge,dt, search_type, use_lorentz, force_velocity_adjust, collision_max, collision_count, collision_pairs, collision_distance))
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
    global electron_velocities, electron_positions, bounds, forces, effective_electron_mass, electron_past_positions, max_velocity

    # Calculate acceleration based on F=ma
    acceleration = forces / effective_electron_mass

    # Update velocities
    electron_velocities += acceleration * dt


    # Limit velocities to max_velocity
    velocity_magnitudes = cp.linalg.norm(electron_velocities, axis=1)
    exceeds_max_velocity = velocity_magnitudes > max_velocity
    # Only apply correction to electrons exceeding max_velocity
    electron_velocities[exceeds_max_velocity] = (electron_velocities[exceeds_max_velocity].T * (max_velocity / velocity_magnitudes[exceeds_max_velocity])).T

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
    global gridx, gridy, gridz, initial_spacing, num_steps, speedup, forces, electron_positions, electron_velocities, dt, initialize_wave

    print("In main")
    if (pulse_width > gridx/2):
        print("pulse_width has to be less than half of gridx")
        exit(-1)

    checkgpu()
    GPUMem()
    if (initialize_wave):
        initialize_electrons_sine_wave()
    else:
        initialize_electrons()


    os.makedirs('simulation', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('velocity', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('density', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('amps', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('speed', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('cappedvelocity', exist_ok=True) # Ensure the simulation directory exists



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
    start_time = time.time()  # Record the start time of the loop
    for step in range(num_steps):
        t = step * dt
        print("In main", step)
        GPUMem()
        if step % wire_steps == 0:            #  
            # WireStatus=calculate_wire_offset(electron_positions)                    # should do this for bound electrons
            # future = client.submit(visualize_wire, "Offset",  WireStatus, step, t)
            density_plot, velocity_plot, amps_plot, speed_plot, cappedvelocity_plot = calculate_plots()
            plots = [
                ("Density", density_plot),
                ("Velocity", velocity_plot),
                ("Amps", amps_plot),
                ("Speed", speed_plot)
                #       ("CappedVelocity", cappedvelocity_plot)
            ]
            # Submit a single future for all plots
            future = client.submit(visualize_all_plots, step, t, plots)
            futures.append(future)
        if step % display_steps == 0:
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

        if (collision_on):                               # see if any new positons violate collision limit if on
            print("resolve collisions", t)
            resolve_collisions()

        cp.cuda.Stream.null.synchronize()         # free memory on the GPU
        elapsed_time = time.time() - start_time           # Calculate elapsed time
        average_time_per_step = elapsed_time / (step + 1) # Calculate the average time per step so far
        estimated_time_left = average_time_per_step * (num_steps - (step + 1)) # Estimate the time left by multiplying the average time per step by the number of steps left
        estimated_completion_time = datetime.now() + timedelta(seconds=estimated_time_left) # Calculate the estimated completion time
        print(f"Step {step + 1}/{num_steps}. Estimated completion time: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()

    copypositions=electron_positions.get()
    copyvelocities=electron_velocities.get()
    future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t) # If we end at 200 we need last output
    futures.append(future)
    wait(futures)
    save_arrays()





if __name__ == '__main__':
    main()
