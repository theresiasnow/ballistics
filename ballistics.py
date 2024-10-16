"""
Ballistics functions for the ballistics calculator.
input: os.environ['AIR_DENSITY'] need to be set to the air density in kg/m^3
"""
import os
from math import atan

import numpy as np
from scipy.integrate import solve_ivp


# Constants
DEFAULT_AIR_DENSITY = 1.225  # Default air density in kg/m^3 at 15°C, sea level
G = 9.82  # Acceleration due to gravity in m/s^2

def get_air_density() -> float:
    air_density_env = os.getenv('AIR_DENSITY')
    if air_density_env == None:
        print("air density not set using default")
        return DEFAULT_AIR_DENSITY
    return float(air_density_env)

def calculate_air_density(temperature = None, pressure=None, humidity=None):
    """
    Calculates air density given temperature (Celsius), pressure (Pa), and humidity (as fraction 0-1).

    :param temperature: Temperature in degrees Celsius
    :param pressure: Atmospheric pressure in Pa
    :param humidity: Relative humidity as a fraction (0 to 1)
    :return: Air density in kg/m^3
    """
    # Constants
    R_d = 287.05  # Specific gas constant for dry air, J/(kg*K)
    R_v = 461.495  # Specific gas constant for water vapor, J/(kg*K)

    def saturation_vapor_pressure(T):
        """
        Calculates the saturation vapor pressure at a given temperature (T in Celsius).
        Formula from: Tetens' formula for saturation vapor pressure.
        """
        # Convert Celsius to Kelvin
        T_kelvin = T + 273.15
        # Tetens formula for vapor pressure in Pa
        p_sat = 6.1078 * 10 ** (7.5 * T / (T + 237.3)) * 100  # Convert from hPa to Pa
        return p_sat

    if temperature is None or pressure is None or humidity is None:
        return DEFAULT_AIR_DENSITY

    # Calculate the saturation vapor pressure at the given temperature
    p_sat = saturation_vapor_pressure(temperature)

    # Partial pressure of water vapor
    p_v = humidity * p_sat

    # Partial pressure of dry air
    p_d = pressure - p_v

    # Convert Celsius to Kelvin
    T_kelvin = temperature + 273.15

    # Calculate air density
    density = (p_d / (R_d * T_kelvin)) + (p_v / (R_v * T_kelvin))

    return density


# Correct drag coefficient calculation to handle unit consistency
# Convert BC from imperial to metric:
def convert_bc_to_metric(bc):
    """
    Convert ballistic coefficient (BC) from imperial to metric units.

    Parameters:
    bc (float): Ballistic coefficient in imperial units.

    Returns:
    float: Ballistic coefficient in metric units.
    """
    # Correct drag coefficient calculation to handle unit consistency
    # Convert BC from imperial to metric:
    inch_to_meter = 0.0254  # Conversion factor from inches to meters
    bc_metric = bc / inch_to_meter ** 2  # Adjust BC for metric units
    return bc_metric

def calculate_velocity_T(v0, t, bc):
    """
    Calculate velocity at time t given initial velocity and ballistic coefficient.

    Parameters:
    v0 (float): Initial velocity in m/s.
    t (float): Time in seconds.
    bc (float): Ballistic coefficient.

    Returns:
    float: Velocity at time t in m/s.
    """
    return v0 * np.exp(-t / bc)

def calculate_drag_coefficient(bc, bullet_weight, bullet_area):  # G1 model for bc
    """
    Calculate the drag coefficient.

    Parameters:
    bc (float): Ballistic coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.

    Returns:
    float: Drag coefficient.
    """
    return bullet_weight / (bc * bullet_area)

def calculate_drag_force(v, drag_coefficient, A):
    """
    Calculate the drag force on a bullet.

    Parameters:
    v (float): Velocity of the bullet in m/s.
    drag_coefficient (float): Drag coefficient.
    A (float): Cross-sectional area of the bullet in m^2.

    Returns:
    float: Drag force in N.
    """
    return 0.5 * drag_coefficient * get_air_density() * v ** 2 * A

def calculate_velocity_at_distance_simple(v0, drag_coefficient, A, d, m):
    """
    Calculate the velocity at a given distance for a simple model.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    A (float): Cross-sectional area of the bullet in m^2.
    d (float): Distance travelled in meters.
    m (float): Mass of the bullet in kg.

    Returns:
    float: Velocity at the given distance in m/s.
    """
    return v0 * np.exp(- (drag_coefficient * get_air_density() * A * d) / (2 * m))

def calculate_barrel_angle(hob, poi, d0):
    """
    Calculate the barrel angle required to hit a target.

    Parameters:
    hob (float): Height over bore in meters.
    poi (float): Point of impact in meters (negative if below the target).
    d0 (float): Distance to the target in meters.

    Returns:
    float: Barrel angle in radians.
    """
    return atan((hob + poi) / d0)  # Angle in radians

def calculate_velocity_at_distance(v0, drag_coefficient, bullet_weight, bullet_area, distance):
    """
    Calculate the velocity at a given distance considering drag.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distance (float): Distance travelled in meters.

    Returns:
    float: Velocity at the given distance in m/s.
    """
    acceleration = -calculate_drag_force(v0, drag_coefficient, bullet_area) / bullet_weight
    return v0 + acceleration * distance / v0

def calculate_velocities(v0, drag_coefficient, bullet_weight, bullet_area, distances):
    """
    Calculate velocities at multiple distances.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distances (list[float]): List of distances in meters.

    Returns:
    list[float]: List of velocities at each distance in m/s.
    """
    velocities = []
    for d in distances:
        velocities.append(calculate_velocity_at_distance(v0, drag_coefficient, bullet_weight, bullet_area, d))
    return velocities

def calculate_velocity(t, y, v0, drag_coefficient, bullet_weight, bullet_area):
    """
    Calculate velocity as a function of time and position.

    Parameters:
    t (float): Time in seconds.
    y (list[float]): List of positions [x, y].
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.

    Returns:
    list[float]: List containing the velocity at the distance travelled.
    """
    distance_travelled = y[0]
    velocity = calculate_velocity_at_distance(v0, drag_coefficient, bullet_weight, bullet_area, distance_travelled)
    return [velocity]

def calculate_time_of_flight(v0, drag_coefficient, bullet_weight, bullet_area, distance, angle_rad):
    """
    Calculate the time of flight to reach a given distance.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distance (float): Distance to the target in meters.
    angle_rad (float): Barrel angle in radians.

    Returns:
    float: Time of flight in seconds.
    """
    if distance == 0:
        return 0
    # Initial conditions: horizontal and vertical velocities
    v_x0 = v0 * np.cos(angle_rad)
    v_y0 = v0 * np.sin(angle_rad)
    y0 = [0, 0, v_x0, v_y0]

    # Estimate max flight time using initial horizontal velocity
    t_max = 2 * distance / v_x0

    # Time steps for numerical integration
    t_eval = np.linspace(0, t_max, 1000)

    # Perform numerical integration using Runge-Kutta method
    sol = solve_ivp(
        bullet_trajectory,
        [0, t_max],
        y0,
        args=(drag_coefficient, bullet_weight, bullet_area),
        t_eval=t_eval,
        method='LSODA',  # Use default Runge-Kutta method
        rtol = 1e-8,  # Relative tolerance for precision
        atol = 1e-10
    )

    # Capture results
    x = sol.y[0]  # Horizontal positions
    y = sol.y[1]  # Vertical positions

    # Find the time when the bullet reaches the target distance
    distances = sol.y[0]
    index = np.argmax(distances >= distance)
    if index == 0:
        raise RuntimeError("Solution did not reach the specified distance.")

    return sol.t[index]


def calculate_time_of_flights(v0, drag_coefficient_g1, bullet_weight, bullet_area, distances, angle_rad):
    """
    Calculate the time of flights to reach multiple distances.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient_g1 (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distances (list[float]): List of distances in meters.
    angle_rad (float): Barrel angle in radians.

    Returns:
    list[float]: List of time of flights to each distance in seconds.
    """
    time_to_distances = []
    for d in distances:
        time_to_distances.append(calculate_time_of_flight(v0, drag_coefficient_g1, bullet_weight, bullet_area, d, angle_rad))
    return time_to_distances

def calculate_poi(d_target, bullet_weight, bullet_area, drag_coefficient, hob, v0, angle_rad):
    """
    Calculate the point of impact (POI) at a given target distance.

    Parameters:
    d_target (float): Target distance in meters.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    drag_coefficient (float): Drag coefficient.
    hob (float): Height over bore in meters.
    v0 (float): Initial velocity in m/s.
    angle_rad (float): Barrel angle in radians.

    Returns:
    float: Final vertical position at the target distance in meters.
    """
    air_density = get_air_density()
    t_total = d_target / (v0 * np.cos(angle_rad))  # Time to reach target distance
    t_steps = np.linspace(0, t_total, 1000)  # Time steps for numerical integration
    x = v0 * np.cos(angle_rad) * t_steps  # Horizontal distance over time
    y = hob + v0 * np.sin(angle_rad) * t_steps - 0.5 * G * t_steps ** 2  # Vertical position without drag
    # Update velocity and position with drag
    for i in range(1, len(t_steps)):
        v_x = v0 * np.cos(angle_rad) * np.exp(- (drag_coefficient * air_density * bullet_area * x[i]) / (2 * bullet_weight))
        v_y = v0 * np.sin(angle_rad) * np.exp(- (drag_coefficient * air_density * bullet_area * x[i]) / (2 * bullet_weight))
        y[i] = hob + v_y * t_steps[i] - 0.5 * G * t_steps[i] ** 2  # Update y position with drag

    return y[-1]  # Return the final vertical position at distance

def calculate_pois(d0, d_max, step, drag_coefficient, hob, v0, angle):
    """
    Calculate the points of impact (POIs) for a range of distances.

    Parameters:
    d0 (int): Initial distance in meters.
    d_max (int): Maximum distance in meters.
    step (int): Step size in meters.
    drag_coefficient (float): Drag coefficient.
    hob (float): Height over bore in meters.
    v0 (float): Initial velocity in m/s.
    angle (float): Barrel angle in radians.

    Returns:
    list[float]: List of points of impact for each distance.
    """
    pois = []
    for i in range(d0, d_max + step, step):
        pois.append(calculate_poi(i, drag_coefficient, hob, v0, angle))
    return pois

# Define the bullet trajectory with drag affecting both x and y velocities
def bullet_trajectory(t, y, drag_coefficient, bullet_weight, bullet_area):
    """
    Define the bullet trajectory with drag affecting both x and y velocities.

    Parameters:
    t (float): Time variable.
    y (list[float]): State vector [x, y, v_x, v_y].
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.

    Returns:
    list[float]: Derivatives [v_x, v_y, drag_x, drag_y - G].
    """
    air_density = get_air_density()
    x, y_pos, v_x, v_y = y
    speed = np.sqrt(v_x ** 2 + v_y ** 2)  # Total speed
    drag_x = -0.5 * drag_coefficient * air_density * bullet_area * speed * v_x / bullet_weight  # Drag force on x
    drag_y = -0.5 * drag_coefficient * air_density * bullet_area * speed * v_y / bullet_weight  # Drag force on y

    return [v_x, v_y, drag_x, drag_y - G]  # Ret

def calculate_poi_with_integration(d_target, drag_coefficient, bullet_weight, bullet_area, hob, v0, angle_rad):
    """
    Calculate the point of impact (POI) at a given target distance using numerical integration.

    Parameters:
    d_target (float): Target distance in meters.
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    hob (float): Height over bore in meters.
    v0 (float): Initial velocity in m/s.
    angle_rad (float): Barrel angle in radians.

    Returns:
    float: Final vertical position at the target distance in meters.
    """
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
        args=(drag_coefficient, bullet_weight, bullet_area),
        t_eval=t_eval,
        method='LSODA',  # More stable for complex problems
        rtol=1e-8,  # Relative tolerance for precision
        atol=1e-10
    )

    x = sol.y[0]  # Horizontal positions
    y = sol.y[1]  # Vertical positions

    return y[-1] - hob  # Difference between final y position and initial height

def calculate_pois_with_integration(d0, d_max, step, drag_coefficient, bullet_weight, bullet_area, hob, v0, angle_rad):
    """
    Calculate points of impact (POIs) for multiple distances using numerical integration.

    Parameters:
    d0 (int): Initial distance in meters.
    d_max (int): Maximum distance in meters.
    step (int): Step size in meters.
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    hob (float): Height over bore in meters.
    v0 (float): Initial velocity in m/s.
    angle_rad (float): Barrel angle in radians.

    Returns:
    np.array: Array of points of impact for each distance.
    """
    pois = []
    for d in range(d0, d_max + step, step):
        poi = calculate_poi_with_integration(d, drag_coefficient, bullet_weight, bullet_area, hob, v0, angle_rad)
        if np.isinf(poi):
            print(f"Warning: Integration failed for distance {d}. POI set to NaN.")
            pois.append(np.nan)
        else:
            pois.append(poi)
    return np.array(pois)

def poi_to_mrad(poi, d):
    """
    Convert point of impact (POI) to milliradians (mrad).

    Parameters:
    poi (float): Point of impact in meters.
    d (float): Distance to the target in meters.

    Returns:
    float: Point of impact in milliradians (mrad).
    """
    if d == 0:
        return 0
    return -(poi / d * 1000.0)

def calculate_mrads(distances, pois):
    """
    Calculate the elevation angle in milliradians (mrad) for multiple distances.

    Parameters:
    distances (list[float]): List of distances in meters.
    pois (list[float]): List of points of impact in meters.

    Returns:
    list[float]: List of elevation angles in milliradians (mrad).
    """
    mrads: list[int | float] = []
    for i in range(len(distances)):
        mrads.append(poi_to_mrad(pois[i], distances[i]))
    return mrads

