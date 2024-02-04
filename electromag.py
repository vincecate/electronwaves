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

simnum=2       # going to have a set to try to get projection to speed of light in full size plane wave

if simnum==1:            # .13% of light speed on Feb 2  
    gridx = 1500         # 
    gridy = 20           # 
    gridz = 20           # 
    speedup = 300        # sort of rushing the simulation time
    pulse_width=600      # how many planes will be given pulse - we simulate half toward middle of this at each end
    pulse_units = 200.5  #
    num_steps =  4000    # how many simulation steps - note dt slows down as this gets bigger unless you adjust speedup
    DisplaySteps = 50    # every so many simulation steps we call the visualize code
    WireSteps = 1        # every so many simulation steps we call the visualize code

if simnum==2:            #
    gridx = 800          # 
    gridy = 40           # 
    gridz = 40           # 
    speedup = 300        # sort of rushing the simulation time
    pulse_width=300      # how many planes will be given pulse - we simulate half toward middle of this at each end
    pulse_units = 100.5  #
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
pulse_offset =pulse_units*initial_spacing    #  how much the first few planes are offset
pulse_speed = 0    # in meters per second 
pulse_sinwave = False  # True if pulse should be sin wave
pulsehalf=False    # True to only pulse half the plane
initialize_orbits=False          # can have electrons initialized to moving if True and not moving if False

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
sim_settings = f"gridx {gridx} gridy {gridy} gridz {gridz} speedup {speedup} Spacing: {initial_spacing:.8e} \n Pulse Width {pulse_width} Speed {pulse_speed:.8e} Units {pulse_units} Steps: {num_steps}"

def GPUMem():
    # Get total and free memory in bytes
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()

    print(f"Total GPU Memory: {total_mem / 1e9} GB")
    print(f"Free GPU Memory: {free_mem / 1e9} GB")

GPUMem()

nucleus_positions = cp.zeros((gridx, gridy, gridz, 3))
electron_positions = cp.zeros((gridx, gridy, gridz, 3))
electron_velocities = cp.zeros((gridx, gridy, gridz, 3))
forces = cp.zeros((gridx, gridy, gridz, 3))

GPUMem()


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

    return(averaged_xdiff.get())    # make Numpy for visualization that runs on CPU


def calculate_histogram(epositions):
    global initial_spacing
    
    # Get x positions 
    x_positions = epositions[:,:,:,0].flatten()
    
    # Convert positions to segment indices
    segment_indices = cp.floor(x_positions / initial_spacing).astype(cp.int32)

    # Calculate histogram
    histogram, _ = cp.histogram(segment_indices, bins=cp.arange(-0.5, gridx + 0.5, 1))

    # Convert the histogram to a NumPy array for visualization
    histogram = histogram.get()

    return histogram


def visualize_wire(averaged_xdiff, step, t):
    # Plotting
    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    #  Want to plot only betweew wire_start and wire_stop
    ax.plot(range(wire_start, wire_stop), averaged_xdiff[wire_start:wire_stop], marker='o')

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
    global pulse_width, pulse_speed, electron_positions, electron_velocities
    yrange=gridy
    if pulsehalf:
        yrange=int(gridy/2)
    for x in range(0,pulse_width):
        if pulse_sinwave:
            offset_add = -1*pulse_offset*np.sin(2*np.pi*x/pulse_width)
            speed_add = -1*pulse_speed*np.sin(2*np.pi*x/pulse_width)
        else:
            offset_add = pulse_offset
            speed_add = pulse_speed
        for y in range(yrange):
            for z in range(gridz):
                electron_positions[x,y,z][0] += offset_add
                electron_velocities[x,y,z][0] += speed_add




def calculate_forces_all():
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
    forces=cp.sum(normforces, axis=1)


#  We are simulating electrons in a wire
#  The grix is large like 3000 and the length of the wire
#  The gridy and gridz are small like 10, 20, or 30 representing the width of the wire
#  The initial position of the electrons is calculated from their grid positon in the electron_positions array.
#  The simulation is assumed to be a short enough time that electrons will not have moved far.
#  If we calculate forces for max_neighbor_grids up/down the X dimension it will be accurate enough.
#  max_neighbor_grids is a global but will start with 50
def calculate_forces_nearby():
    global electron_positions, forces, gridx, gridy, gridz, max_neighbor_grids, coulombs_constant, electron_charge

    # Reset forces to zero before calculation
    forces.fill(0)

    # Loop through all electrons by their grid indices
    for x in range(gridx):
        for y in range(gridy):
            for z in range(gridz):
                # Calculate range of x indices to consider based on max_neighbor_grids
                x_start = max(0, x - max_neighbor_grids)
                x_end = min(gridx, x + max_neighbor_grids + 1)  # +1 because range end is exclusive
                
                # Accumulator for forces on the current electron
                force_on_current = cp.zeros(3)
                
                # Loop through nearby electrons in the x direction and all y, z
                for xi in range(x_start, x_end):
                    for yi in range(gridy):
                        for zi in range(gridz):
                            if xi == x and yi == y and zi == z:
                                continue  # Skip self-interaction
                            
                            # Compute distance vector between current electron and others
                            delta_r = electron_positions[x, y, z] - electron_positions[xi, yi, zi]
                            distance = cp.linalg.norm(delta_r)
                            if distance == 0:  # Prevent division by zero
                                continue
                            
                            # Calculate force magnitude using Coulomb's law
                            force_magnitude = coulombs_constant * (electron_charge ** 2) / (distance ** 2)
                            
                            # Calculate force vector and add it to the accumulator
                            force_on_current += (delta_r / distance) * force_magnitude
                
                # Update the forces array with the calculated force
                forces[x, y, z] = force_on_current



def calculate_forces_chunked():
    global electron_positions, forces, gridx, gridy, gridz, max_distance
    forces.fill(0)  # Reset forces to zero

    # Number of electrons
    num_electrons = gridx * gridy * gridz

    # Calculate chunk size based on available memory and desired batch size
    # Adjust `batch_size` based on your GPU's memory capacity
    batch_size = 5000  # Example batch size, adjust based on your system's capacity

    for start_idx in range(0, num_electrons, batch_size):
        end_idx = min(start_idx + batch_size, num_electrons)

        # Extract batch positions
        batch_positions = electron_positions.reshape(-1, 3)[start_idx:end_idx]

        # Compute distances and forces between the batch and all electrons
        for other_start in range(0, num_electrons, batch_size):
            other_end = min(other_start + batch_size, num_electrons)

            # Extract positions of other electrons to compare with the current batch
            other_positions = electron_positions.reshape(-1, 3)[other_start:other_end]

            # Calculate pairwise differences and distances
            delta_r = batch_positions[:, None, :] - other_positions[None, :, :]
            distances = cp.sqrt(cp.sum(delta_r**2, axis=2))

            # Apply max distance constraint
            mask = (distances < max_distance * initial_spacing) & (distances > 0)  # Exclude self-interactions
            distances[~mask] = cp.inf

            # Calculate forces
            force_magnitude = coulombs_constant * (electron_charge**2) / distances**2
            force_direction = delta_r / distances[..., None]

            # Sum forces and update the forces array
            total_forces = cp.sum(force_magnitude[..., None] * force_direction, axis=1)

            # Update forces for the current batch
            forces.reshape(-1, 3)[start_idx:end_idx] += total_forces

    # Reshape forces back to the original shape
    forces = forces.reshape(gridx, gridy, gridz, 3)





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
    update_mask = (x_indices >= sim_start) & (x_indices < sim_stop)
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
    GPUMem()
    initialize_atoms()
    os.makedirs('simulation', exist_ok=True) # Ensure the simulation directory exists


    # Create a LocalCluster with a custom death timeout and then a Client
    cluster = LocalCluster(n_workers=24, death_timeout='600s')
    client= Client(cluster)
    futures = []

    print("Doing first visualization")
    copypositions=electron_positions.get()   # get makes Numpy copy so runs on CPU in Dask
    copyvelocities=electron_velocities.get() # get makes Numpy copy so runs on CPU in Dask
    future = client.submit(visualize_atoms, copypositions, copyvelocities, -1, 0.0)
    futures.append(future)
    print("Doing pulse")
    pulse()
    # main simulation loop
    for step in range(num_steps):
        t = step * dt
        print("In main", step)
        GPUMem()
        if step % WireSteps == 0:
            # WireStatus=calculate_wire(electron_positions)
            WireStatus=calculate_histogram(electron_positions)
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
        cp.cuda.Stream.null.synchronize()         # free memory on the GPU

    copypositions=electron_positions.get()
    copyvelocities=electron_velocities.get()
    future = client.submit(visualize_atoms, copypositions, copyvelocities, step, t) # If we end at 200 we need last output
    futures.append(future)
    wait(futures)



if __name__ == '__main__':
    main()
