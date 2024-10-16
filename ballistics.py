## Ballistics functions for the ballistics calculator
from math import atan
import numpy as np
from scipy.integrate import solve_ivp

g = 9.82  # Acceleration due to gravity in m/s^2

# Correct drag coefficient calculation to handle unit consistency
# Convert BC from imperial to metric:
def convert_bc_to_metric(bc):
    inch_to_meter = 0.0254  # Conversion factor from inches to meters
    bc_metric = bc / inch_to_meter ** 2  # Adjust BC for metric units
    return bc_metric

def calculate_velocity_T(v0, t, bc):
    return v0 * np.exp(-t / bc)

def calculate_drag_coefficient(bc, bullet_weight, bullet_area):  # G1 model for bc
    return bullet_weight / (bc * bullet_area)

def calculate_drag_force(v, air_density, drag_coefficient, A):
    return 0.5 * drag_coefficient * air_density * v ** 2 * A

def calculate_velocity_at_distance(v0, air_density, drag_coefficient, A, d, m):
    return v0 * np.exp(- (drag_coefficient * air_density * A * d) / (2 * m))

def calculate_barrel_angle(hob, poi, d0):
    # hob: height over bore in meters
    # poi: point of impact in meters (negative if below the target)
    # distance: distance to the target in meters
    return atan((hob + poi) / d0)  # Angle in radians

# TODO refactor
# Function to calculate time of flight considering initial velocity and drag
def calculate_time_of_flight(v0, air_density, bullet_weight, bullet_area, drag_coefficient, d):
    # Define a small-time step for integration
    dt = 0.001  # time step in seconds
    time = 0.0  # initialize time
    distance_travelled = 0.0  # initialize distance
    # Iterate until the bullet reaches the target distance
    while distance_travelled < d:
        velocity = calculate_velocity_at_distance(v0, air_density, drag_coefficient, bullet_area, distance_travelled, bullet_weight)
        distance_travelled += velocity * dt
        time += dt
    return time

def calculate_velocities(v0, air_density, drag_coefficient, bullet_weight, bullet_area, distances):
    velocities = []
    for d in distances:
        velocities.append(calculate_velocity_at_distance(v0, air_density, drag_coefficient, bullet_area, d, bullet_weight))
    return velocities


def calculate_poi(d_target, air_density, bullet_weight, bullet_area, drag_coefficient, hob, v0, angle_rad):
    t_total = d_target / (v0 * np.cos(angle_rad))  # Time to reach target distance
    t_steps = np.linspace(0, t_total, 1000)  # Time steps for numerical integration
    x = v0 * np.cos(angle_rad) * t_steps  # Horizontal distance over time
    y = hob + v0 * np.sin(angle_rad) * t_steps - 0.5 * g * t_steps ** 2  # Vertical position without drag

    # Update velocity and position with drag
    for i in range(1, len(t_steps)):
        v_x = v0 * np.cos(angle_rad) * np.exp(- (drag_coefficient * air_density * bullet_area * x[i]) / (2 * bullet_weight))
        v_y = v0 * np.sin(angle_rad) * np.exp(- (drag_coefficient * air_density * bullet_area * x[i]) / (2 * bullet_weight))
        y[i] = hob + v_y * t_steps[i] - 0.5 * g * t_steps[i] ** 2  # Update y position with drag

    return y[-1]  # Return the final vertical position at distance


def calculate_pois(d0, d_max, step, drag_coefficient, hob, v0, angle):
    pois = []
    for i in range(d0, d_max + step, step):
        pois.append(calculate_poi(i, drag_coefficient, hob, v0, angle))
    return pois

# Define the bullet trajectory with drag affecting both x and y velocities
def bullet_trajectory(t, y, air_density, drag_coefficient, bullet_weight, bullet_area):
    x, y_pos, v_x, v_y = y
    speed = np.sqrt(v_x ** 2 + v_y ** 2)  # Total speed
    drag_x = -0.5 * drag_coefficient * air_density * bullet_area * speed * v_x / bullet_weight  # Drag force on x
    drag_y = -0.5 * drag_coefficient * air_density * bullet_area * speed * v_y / bullet_weight  # Drag force on y

    return [v_x, v_y, drag_x, drag_y - g]  # Ret

def calculate_poi_with_integration(d_target, air_density, drag_coefficient, bullet_weight, bullet_area, hob, v0, angle_rad):
    if d_target == 0:
        # For zero distance, return the initial height above ground level
        return np.nan
    v_x0 = v0 * np.cos(angle_rad)  # Initial velocity component in the x direction
    v_y0 = v0 * np.sin(angle_rad)  # Initial velocity component in the y direction
    y0 = [0, hob, v_x0, v_y0]  # Initial conditions: [x0, y0, vx0, vy0]
    t_max = d_target / v_x0
    # Ensure t_eval is a 1-dimensional array
    t_eval = np.linspace(0, t_max, 1000)

    # Use adaptive integration (Runge-Kutta method)
    # sol = solve_ivp(bullet_trajectory, [0, t_max], y0, args=(drag_coefficient, bullet_weight), t_eval=t_eval)

    sol = solve_ivp(
        bullet_trajectory,
        [0, t_max],
        y0,
        args=(air_density, drag_coefficient, bullet_weight, bullet_area),
        t_eval=t_eval,
        method='LSODA',  # More stable for complex problems
        rtol=1e-8,  # Relative tolerance for precision
        atol=1e-10
    )

    x = sol.y[0]  # Horizontal positions
    y = sol.y[1]  # Vertical positions

    return y[-1] - hob  # Difference between final y position and initial height

def calculate_pois_with_integration(d0, d_max, step, air_density, drag_coefficient, bullet_weight, bullet_area, hob, v0, angle_rad):
    pois = []
    for d in range(d0, d_max + step, step):
        poi = calculate_poi_with_integration(d, air_density, drag_coefficient, bullet_weight, bullet_area, hob, v0, angle_rad)
        if np.isinf(poi):
            print(f"Warning: Integration failed for distance {d}. POI set to NaN.")
            pois.append(np.nan)
        else:
            pois.append(poi)
    return np.array(pois)

def poi_to_mrad(poi, d):
    if d == 0:
        return 0
    return -(poi / d * 1000.0)

def calculate_mrads(distances, pois):
    mrads = []
    for i in range(len(distances)):
        mrads.append(poi_to_mrad(pois[i], distances[i]))
    return mrads
