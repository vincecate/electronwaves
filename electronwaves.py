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
from scipy.constants import e, epsilon_0, electron_mass, elementary_charge, c, Boltzmann    
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
initialize_wave = sim_settings.get('initialize_wave', False)   # Try to initialize in a wave pattern so not in rush to move 
pulse_velocity = sim_settings.get('pulse_velocity', 0)   # Have electrons in pulse area moving
pulse_offset = sim_settings.get('pulse_offset', 0)   # X value offset for pulse 
force_velocity_adjust = sim_settings.get('force_velocity_adjust', True)   # X value offset for pulse 
collision_distance = sim_settings.get('collision_distance', 1e-10)  # Less than this simulate a collision
ee_collisions_on = sim_settings.get('ee_collisions_on', True)  # Simulate electron electon collisions 
collision_max = sim_settings.get('collision_max', 1000) # Maximum number of collisions per time slice
driving_current = sim_settings.get('driving_current', 0.0) # Amps applied to wire 
driving_voltage = sim_settings.get('driving_voltage', 0.0) # Don't have this yet but idea is to keep certain number of electrons in slice
driving_end_perc = sim_settings.get('driving_end_perc', 5) # 5 percent of both ends for taking and adding elect
latice_collisions_on = sim_settings.get('latice_collisions_on', True) # 
mean_free_path = sim_settings.get('mean_free_path', 4e-8) #  Seems for drif velocity 4e-8 works but for Fermi velocity may not be right
amps_method = sim_settings.get('amps_method', 1) #  Used in calculate_plots:  1=use electron_past_positions  2=use drift velocity and counts

effective_electron_mass = electron_mass   #  default is the same
# Initial electron speed 2,178,278 m/s
# electron_speed= 2178278  
electron_speed= 2188058
electrons_per_coulomb = 1.0 / abs(elementary_charge)
coulombs_per_electron = abs(elementary_charge)

# Atom spacing in meters
hydrogen_spacing = 3.34e-9  # 3.34 nanometers between atoms in hydrogen gas
copper_spacing = 0.128e-9  # 3.34 nanometers between atoms in copper solid
initial_spacing_gemini = 2.27e-10  # Gemini Ultra number for free electron spacing
initial_spacing = copper_spacing*47  # 47^3 is about 100,000 and 1 free electron for every 100,000 copper atoms
initial_radius = 5.29e-11 #  initial electron radius for hydrogen atom - got at least two times
pulse_sinwave = False  # True if pulse should be sin wave


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
sim_settings = f"simnum {simnum} gridx {gridx} gridy {gridy} gridz {gridz} speedup {speedup} lorentz {use_lorentz} ee-col {ee_collisions_on} amps {amps_method} \n latice {latice_collisions_on} mfp {mean_free_path:.4e} driving_c {driving_current:.4e} Spacing: {initial_spacing:.4e} PWidth {pulse_width} PDensity {pulse_density} PVeloc {pulse_velocity} Steps: {num_steps} dt: {dt:.4e} iv:{initialize_velocities} st:{search_type}"

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
electron_is_active = cp.zeros(num_electrons, dtype=bool)   # To take electrons off the right end and add to left
electron_positions = cp.zeros((num_electrons, 3))
electron_velocities = cp.zeros((num_electrons, 3))
forces = cp.zeros((num_electrons, 3))
#electron_is_bound = cp.zeros(num_electrons, dtype=bool)   # if True then electron is bound to an atom, if flase then free
#electron_atom_center = cp.zeros((num_electrons, 3))       # atom position that spring for bound electron is attached to

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



# Note  - boltz_temp has temp in K like 300
# Main array globals were define as:
# electron_positions = cp.zeros((num_electrons, 3))
# electron_velocities = cp.zeros((num_electrons, 3))
# Other globals used here are all constants defined at start
# Want probability of collision to depend on velocity and mean_free_path
#   velocity after collision just on boltz_temp
# Arrays are in CuPy so want to do vector calculations
# dt is length of simulation timestep in seconds
def latice_collisions():
    global electron_positions, electron_velocities
    global num_electrons, boltz_temp, Boltzmann, electron_mass, mean_free_path, dt

    # Calculate velocities magnitude
    velocities_magnitude = cp.sqrt(cp.sum(electron_velocities ** 2, axis=1))

    # Adjusted collision probabilities incorporating dt
    mean_free_time = mean_free_path / velocities_magnitude
    adjusted_probabilities = 1 - cp.exp(-dt / mean_free_time)

    # Decide which electrons collide
    random_numbers = cp.random.rand(num_electrons)
    collide = random_numbers < adjusted_probabilities


    # Count the number of collisions
    num_collisions = cp.sum(collide)

    # Update velocities after collision using a simple model
    # For a more accurate model, you might simulate the actual direction and magnitude based on boltz_temp
    thermal_velocity = cp.sqrt(2 * Boltzmann * boltz_temp / electron_mass)
    direction = cp.random.normal(size=(num_electrons, 3))
    direction /= cp.linalg.norm(direction, axis=1)[:, cp.newaxis]  # Normalize to get direction vectors
    electron_velocities[collide] = direction[collide] * thermal_velocity

    print(f"latice_collisions did {num_collisions} collisions ")

# Note  - CUDA kerel has made a list of collision_pairs that is collision_count long
# electron_positions = cp.zeros((num_electrons, 3))
# electron_velocities = cp.zeros((num_electrons, 3))
def resolve_ee_collisions():
    global electron_positions, electron_velocities, collision_count, collision_pairs, collision_distance

    # Read the number of collisions detected
    num_collisions = collision_count.item()  # Convert to a Python scalar

    # Read the collision pairs, and slice based on the actual number of collisions
    collision_pairs_np = collision_pairs[:num_collisions].get()

    print(f"Number of e-e collisions: {num_collisions}")
    print(collision_pairs_np)

    desired_separation = 2 * collision_distance  # The target separation distance

    for i in range(len(collision_pairs_np)):
        e1, e2 = collision_pairs_np[i]
        collision_vector = electron_positions[e1] - electron_positions[e2]
        norm_collision_vector = cp.linalg.norm(collision_vector)
        collision_vector_normalized = collision_vector / norm_collision_vector

        v1 = electron_velocities[e1]
        v2 = electron_velocities[e2]

        # Calculate the projections of the velocities onto the collision vector
        v1_proj = cp.dot(v1, collision_vector_normalized)
        v2_proj = cp.dot(v2, collision_vector_normalized)

        # Swap the velocity components along the collision vector (elastic collision)
        v1_new = v1 - v1_proj * collision_vector_normalized + v2_proj * collision_vector_normalized
        v2_new = v2 - v2_proj * collision_vector_normalized + v1_proj * collision_vector_normalized

        electron_velocities[e1] = v1_new
        electron_velocities[e2] = v2_new

        # Adjust positions to ensure they are more than the desired_separation apart
        separation_correction = (desired_separation - norm_collision_vector) / 2
        electron_positions[e1] += separation_correction * collision_vector_normalized
        electron_positions[e2] -= separation_correction * collision_vector_normalized

    # Zero out collision count for the next iteration
    collision_count.fill(0)




def generate_thermal_velocities():
    global num_electrons, boltz_temp, Boltzmann, electron_mass
    """
    Generates random thermal velocity vectors for a given number of electrons
    at a specified temperature. The Maxwell-Boltzmann distribution is used to
    determine the magnitudes of the velocities.

    Returns:
        cupy.ndarray: An array of shape (num_electrons, 3) containing random
                      velocity vectors for each electron.
    """

    # Calculate the standard deviation of the speed distribution
    sigma = cp.sqrt(Boltzmann * boltz_temp / electron_mass)

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
    adjusted_pulse_electrons = int(num_electrons * pulse_volume_ratio * pulse_density)  # higher density in pulse region
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

   

def initialize_past():
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

    electron_is_active.fill(True)                       # for now all electrons are active


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




# electron_positions = cp.zeros((num_electrons, 3))
# electron_velocities = cp.zeros((num_electrons, 3))
# past_positions_count = 100
# electron_past_positions = cp.zeros((num_electrons, past_positions_count, 3))   # keeping past positions with current at 0, previos 1, etc
#  returns  (density, velocity, amps, speed )    - for plotting to files
#  For amps we use past_segment_indices to could up electrons moved right or left
#   electron_past_positions[0] is current position and [1] is previous time slice position, etc.
#    Each slice of the wire is initial_spacing wide.
def calculate_plots():
    global electron_positions, electron_velocities, gridx, gridy, gridz, initial_spacing, electron_charge, dt

    # Get x positions and velocities from the 2D arrays
    x_positions = electron_positions[:, 0]
    past_x_positions = electron_past_positions[:,1,0]     #  past[0] is current and past[1] is one time slice back

    print(f"plots x-12 {x_positions[12]} past-x-12 {past_x_positions[12]}")

    x_velocities = electron_velocities[:, 0]

    # Convert current positions and past positions to segment indices based on the total length of the wire and number of slices
    segment_indices = cp.floor(x_positions / initial_spacing).astype(cp.int32)
    past_segment_indices = cp.floor(past_x_positions / initial_spacing).astype(cp.int32)

    # Ensure segment indices are within bounds
    segment_indices = cp.clip(segment_indices, 0, gridx - 1)
    past_segment_indices = cp.clip(past_segment_indices, 0, gridx - 1)

    # Calculate speeds (magnitude of velocity) for each electron
    speeds = cp.sqrt(cp.sum(electron_velocities**2, axis=1))

    # Use bincount to sum xvelocities, sum speeds, and count electrons in each segment
    xvelocity_sums = cp.bincount(segment_indices, weights=x_velocities, minlength=gridx)
    speed_sums = cp.bincount(segment_indices, weights=speeds, minlength=gridx)
    electron_counts = cp.bincount(segment_indices, minlength=gridx)
    # Avoid division by zero
    electron_counts_nonzero = cp.where(electron_counts == 0, 1, electron_counts)

    # Calculate average velocities (drift_velocity)  and average speeds
    drift_velocities = xvelocity_sums / electron_counts_nonzero
    average_speeds = speed_sums / electron_counts_nonzero

    if (amps_method == 1):
        # Calculate the movement direction of electrons between segments
        moved_directions = cp.sign(segment_indices - past_segment_indices)
        moved_counts = cp.bincount(segment_indices, weights=moved_directions, minlength=gridx)

        # Calculate the current flowing through each segment
        amps = moved_counts * coulombs_per_electron / dt
        amps[0]=0                          # way we do it can't get value for 0
        if driving_current > 0:  # if we are taking electrons from the end 
            endzone = int(gridx * (100.0 - driving_end_perc) / 100.0)
            for i in range(endzone, gridx):
                amps[i] = 0             #  then the amps at the end is no good
    else:   # amps_method == 2
        # double fraction_moved = drift_velocities * dt / initial_spacing    # These two lines logically what we are doing
        # amps = fraction_moved * electron_counts * coulombs_per_electron/dt #   but since dt muliplied and then divided
        fraction_moved_per_dt = drift_velocities / initial_spacing           # These two lines have dt factored out 
        amps = fraction_moved_per_dt * electron_counts * coulombs_per_electron      #   to run faster 
        

    print(f"drift-50 {drift_velocities[50]} density-50 {electron_counts[50]} amps-50 {amps[50]}")

    return electron_counts.get(), drift_velocities.get(), amps.get(), average_speeds.get()







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



# Load CUDA source and get kernel function
with open('kernel.cu', 'r') as file:
    kernel_code = file.read()
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
    global electron_positions, electron_velocities, electron_past_positions, electron_past_velocities, past_positions_count, forces, num_electrons, coulombs_constant, electron_charge, dt, search_type, use_lorentz, force_velocity_adjust, collision_max, collision_count, collision_pairs, collision_distance, electron_is_active
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
                     args=(electron_positions, electron_velocities, electron_past_positions, electron_past_velocities, past_positions_count, forces, num_electrons, coulombs_constant, electron_charge,dt, search_type, use_lorentz, force_velocity_adjust, collision_max, collision_count, collision_pairs, collision_distance, electron_is_active))
        cp.cuda.Device().synchronize()    #  Let this finish before we do anything else

        end_gpu.record()
        end_gpu.synchronize()
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        print(f"Calculate_forces duration {t_gpu/1000} seconds")

    except cp.cuda.CUDARuntimeError as e:
        print(f"CUDA Error: {e}")
    print("ending calculate_forces_cuda")

# Note gobal arrays are defined as:
#electron_is_active = cp.zeros(num_electrons, dtype=bool)   # To take electrons off the right end and add to left
#electron_positions = cp.zeros((num_electrons, 3))
#electron_velocities = cp.zeros((num_electrons, 3))
#float driving_current
#float dt       # float time interval for simulation
#
# This is called once per dt time.
# We want to remove enough electrons from the right and add to the left
# to make the amps for this equal to driving_current for this dt.
# The wire is gridx units long and gridy and gridz high/wide each unit being initial_spacing.
# We use CuPy to find electrons in the last part of the wire and then 
#  randomly pick enough to move to the first part of the wire.
#  We can initialize the velocity after the move to zero.
def apply_driving_current():
    global electron_positions, electron_velocities, electron_is_active, num_electrons, electron_charge, dt
    global driving_end_perc

    # Constants
    volume_of_wire = gridx * gridy * gridz * (initial_spacing ** 3)  # Volume of the wire
    density_of_electrons = num_electrons / volume_of_wire  # Electron density
    current_density = driving_current / (gridy * gridz * (initial_spacing ** 2))  # Current density (A/m^2)

    # The required change in the number of electrons based on the current and time interval
    coulombs_per_dt = driving_current * dt                               # Amp is C/second, so times dt gives units of coulombs 
    electrons_per_dt = int(cp.floor(coulombs_per_dt * electrons_per_coulomb ))  # want int number of electrons to move

    # Identify electrons in the last part of the wire
    end_in_meters = initial_spacing * gridx * (100.0 - driving_end_perc) / 100.0
    start_in_meters = initial_spacing * gridx * driving_end_perc / 100.0
    right_end_electrons = electron_positions[:, 0] >= end_in_meters 
    indices_of_electrons_to_move = cp.where(right_end_electrons)[0]
    
    # Check if there are any electrons to move
    if indices_of_electrons_to_move.size > 0:
        if len(indices_of_electrons_to_move) > electrons_per_dt:
            # If more electrons are available than needed, randomly select a subset
            chosen_indices = cp.random.choice(indices_of_electrons_to_move, size=electrons_per_dt, replace=False)
        else:
            # If fewer electrons are available than needed, move all of them
            chosen_indices = indices_of_electrons_to_move
        
        # Move selected electrons to the start of the wire
        # Randomly select new positions between 0.1 * initial_spacing and initial_spacing for each electron
        new_positions = cp.random.uniform(0 , start_in_meters, size=len(chosen_indices))
        electron_positions[chosen_indices, 0] = new_positions  # Update positions with random values in the specified range
        electron_velocities[chosen_indices] = 0  # Reset their velocities to 0

        print(f"{len(chosen_indices)} electrons moved from right to left to apply driving current.")
    else:
        print("No electrons in the rightmost segment to move.")



  

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

    print("Wire length in mm is ",gridx*initial_spacing*1000)

    checkgpu()
    GPUMem()
    if (initialize_wave):
        initialize_electrons_sine_wave()
    else:
        initialize_electrons()

    load_arrays()        #   we are not saving past_positions to file so far - if no filename_load "none" it  will not change arrays
    initialize_past()    #  past_electron_positions and past_electron_velocities
    electron_is_active.fill(True)                       # for now all electrons are active

    os.makedirs('simulation', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('velocity', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('density', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('amps', exist_ok=True) # Ensure the simulation directory exists
    os.makedirs('speed', exist_ok=True) # Ensure the simulation directory exists



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
            density_plot, velocity_plot, amps_plot, speed_plot = calculate_plots()
            plots = [
                ("Density", density_plot),
                ("Velocity", velocity_plot),
                ("Amps", amps_plot),
                ("Speed", speed_plot)
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

        if (driving_current > 0):                 # we move some electrons from right edge of wire to left
            print("apply_driving_current")        # do this before checking for collisions
            apply_driving_current()

        if (ee_collisions_on):                        # see if any new positons violate collision limit if on
            print("resolve_ee_collisions")
            resolve_ee_collisions()

        if (latice_collisions_on):                # does not change electron positions just velocity
            print("latice_collisions")
            latice_collisions()

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
