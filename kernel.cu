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
                                double collision_distance,
                                bool* electron_is_active) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double speed_of_light = 299792458.0; // Speed of light in meters per second

    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[CUDA-Cupy] num_electrons %d\\n", num_electrons );  // print once per kernel launch

    if (i < num_electrons && electron_is_active[i]) {
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
                    if (i < j){                     // only smaller of pair has to record collision
                        int collision_id = atomicAdd(collision_count, 1);
                        if (collision_id < collision_max) {
                            collision_pairs[2 * collision_id] = i;           // record this electron
                            collision_pairs[2 * collision_id + 1] = j;       // and one we collide with
                        }
                    }
                    force.x = 0.0;      // in collision we don't calc force as can be nan trouble
                    force.y = 0.0;
                    force.z = 0.0;
                } else {
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

                    if (isnan(coulomb)){
                        printf("found nan at i=%d   j=%d \\n",i,j);
                    } else{
                        force.x += coulomb * normalized_r.x;
                        force.y += coulomb * normalized_r.y;
                        force.z += coulomb * normalized_r.z;
                    }
                }  // if collision or else
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

