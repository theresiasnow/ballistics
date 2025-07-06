"""
Ballistics functions for the ballistics calculator.
input: os.environ['AIR_DENSITY'] need to be set to the air density in kg/m^3
"""
import math
from math import atan, cos, radians, pi

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import util as util

# Constants
INCHES_TO_METERS_FACTOR = 0.0254
KMH_TO_MPS = 1000 / 3600  # Conversion factor from km/hob to m/s
G = 9.82  # Acceleration due to gravity in m/s^2

def bullet_dynamics(t, y, drag_coefficient, bullet_mass, bullet_area, wind_speed=0.0, wind_angle=0.0):
    """
    Compute the time derivatives of position and velocity for the bullet.

    y: Array of [x, y, vx, vy] where:
        x: horizontal position (m)
        y: vertical position (m)
        vx: horizontal velocity (m/s)
        vy: vertical velocity (m/s)

    Returns: derivatives [vx, vy, ax, ay] where:
        ax: horizontal acceleration (m/s^2)
        ay: vertical acceleration (m/s^2)
    """
    x, y_pos, z, vx, vy, vz = y

    # Wind components in m/s (wind direction relative to bullet's path)
    wind_vx = wind_speed * np.cos(wind_angle)
    wind_vz = wind_speed * np.sin(wind_angle)

    # Relative velocities in m/s
    relative_vx = vx - wind_vx
    relative_vy = vy  # Assuming no vertical wind component
    relative_vz = vz - wind_vz

    # Total relative velocity magnitude in m/s
    relative_velocity = np.sqrt(relative_vx ** 2 + relative_vy ** 2 + relative_vz ** 2)

    # Drag force in Newtons (N)
    # drag_force = 0.5 * drag_coefficient * bullet_area * air_density * relative_velocity ** 2
    drag_force = calculate_drag_force(relative_velocity, drag_coefficient, bullet_area)

    # Accelerations in m/s²
    ax = -(drag_force * relative_vx) / (bullet_mass * relative_velocity)
    ay = -G - (drag_force * relative_vy) / (bullet_mass * relative_velocity)
    az = -(drag_force * relative_vz) / (bullet_mass * relative_velocity)

    return [vx, vy, vz, ax, ay, az]

def calibrate_zero(v0, d_zero, drag_coefficient_g1, bullet_weight, bullet_area, hob):
    """
    Calibrate the barrel angle to achieve a zero point at a specified distance.

    Parameters:
    v0 (float): Initial velocity of the bullet in m/s.
    d_zero (float): Distance to the zero point in meters.
    drag_coefficient_g1 (float): Drag coefficient of the bullet (G1 model).
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    hob (float): Height over bore in meters.

    Returns:
    float: The calibrated barrel angle in radians.
    """
    print("Calibrating zero...")
    # Initial guess for the angle
    angle_guess = 0

    # Function to find the angle that makes y(d_target) = 0 (hit the target at height zero)
    def find_angle(a):
        # a is passed as an array, so we extract the scalar value using a[0]
        a = a[0]
        #    return b.calculate_poi(v0, d_zero, drag_coefficient_g1, bullet_mass, bullet_area, hob, a)
        return calculate_poi(v0, d_zero, drag_coefficient_g1, bullet_weight, bullet_area, hob, a)

    # Use fsolve to find the correct angle
    result = fsolve(find_angle, angle_guess)
    print("Calibration complete. Barrel angle:", result[0])
    return result[0]

def calculate_air_density(temperature: float = None, pressure: float = None, humidity: float = None) -> float:
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

    def saturation_vapor_pressure(T: float) -> float:
        """
        Calculates the saturation vapor pressure at a given temperature (T in Celsius).
        Formula from: Tetens' formula for saturation vapor pressure.
        """
        # Tetens formula for vapor pressure in Pa
        return 6.1078 * 10 ** (7.5 * T / (T + 237.3)) * 100  # Convert from hPa to Pa

    if temperature is None or pressure is None or humidity is None:
        return util.DEFAULT_AIR_DENSITY

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

def calculate_true_ballistic_range(distance, angle, k=0.0):
    """
    Calculate the true ballistic range considering the angle and an optional constant.

    Parameters:
    distance (float): The horizontal distance in meters.
    angle (float): The angle of elevation in degrees.
    k (float, optional): An optional constant to adjust the range. Default is 0.0.

    Returns:
    float: The true ballistic range in meters.
    """
    return distance * cos(radians(angle)) + k

def calculate_true_ballistic_ranges(distances, angle, k=0.0):
    """
    Calculate the true ballistic ranges for multiple distances considering the angle and an optional constant.

    Parameters:
    distances (list[float]): List of horizontal distances in meters.
    angle (float): The angle of elevation in degrees.
    k (float, optional): An optional constant to adjust the range. Default is 0.0.

    Returns:
    np.array: Array of true ballistic ranges in meters.
    """
    return np.array([calculate_true_ballistic_range(d, angle, k) for d in distances])

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
    inch_to_meter = INCHES_TO_METERS_FACTOR  # Conversion factor from inches to meters
    bc_metric = bc / inch_to_meter ** 2  # Adjust BC for metric units
    return bc_metric

def calculate_drag_coefficient(bc, bullet_mass, bullet_area):  # G1 model for bc
    """
    Calculate the drag coefficient.

    Parameters:
    bc (float): Ballistic coefficient.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.

    Returns:
    float: Drag coefficient.
    """
    return bullet_mass / (bc * bullet_area)

def calculate_drag_force(v, drag_coefficient, bullet_area):
    """
    Calculate the drag force on a bullet.

    Parameters:
    v (float): Velocity of the bullet in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    Returns:
    float: Drag force in N.
    """
    return 0.5 * drag_coefficient * util.get_air_density() * v ** 2 * bullet_area

def calculate_barrel_angle(hob, poi, d0):
    """
    Calculate the barrel angle required to hit a target.

    Parameters:
    hob (float): Height over bore in meters.
    k (float): Point of impact in meters (negative if below the target).
    d0 (float): Distance to the target in meters.

    Returns:
    float: Barrel angle in radians.
    """
    return atan((hob + poi) / d0)  # Angle in radians

def calculate_velocity_at_distance(v0, drag_coefficient, bullet_mass, bullet_area, distance, angle):
    """
    Calculate the velocity at a given distance using numerical integration.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distance (float): Distance to the target in meters.
    angle (float): Barrel angle in radians.

    Returns:
    float: Velocity at the specified distance in m/s."""

    if distance == 0:
        return v0

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction (0 for horizontal shot)
    v0z = 0  # Initial velocity in z-direction (no initial drift)
    y0 = [0, 0, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    # Increase the estimated time of flight
    t_max = calculate_time_of_flight(v0, drag_coefficient, bullet_mass, bullet_area, distance, angle) * 2
    t_span = [0, t_max]

    sol = solve_ivp(
        bullet_dynamics,
        t_span,
        y0,
        args=(drag_coefficient, bullet_mass, bullet_area),
        method='RK45',
        dense_output=True,
        rtol=1e-8,
        atol=1e-10
    )

    x_vals = sol.y[0]  # Horizontal positions
    vx_vals = sol.y[3]  # Horizontal velocities
    vy_vals = sol.y[4]  # Vertical velocities
    vz_vals = sol.y[5]  # Z-direction velocities

    index = np.argmax(x_vals >= distance)

    if index == 0:
        raise RuntimeError("Bullet did not reach the specified distance.")

    x_before = x_vals[index - 1]
    x_after = x_vals[index]
    vx_before = vx_vals[index - 1]
    vx_after = vx_vals[index]
    vy_before = vy_vals[index - 1]
    vy_after = vy_vals[index]
    vz_before = vz_vals[index - 1]
    vz_after = vz_vals[index]

    interpolate = lambda before, after: before + (distance - x_before) * (after - before) / (x_after - x_before)

    vx_target = interpolate(vx_before, vx_after)
    vy_target = interpolate(vy_before, vy_after)
    vz_target = interpolate(vz_before, vz_after)

    final_velocity = np.sqrt(vx_target ** 2 + vy_target ** 2 + vz_target ** 2)

    return final_velocity


def calculate_velocities(v0, drag_coefficient, bullet_mass, bullet_area, distances, angle):
    """
    Calculate velocities at multiple distances.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distances (list[float]): List of distances in meters.

    Returns:
    list[float]: List of velocities at each distance in m/s.
    """
    velocities = []
    for d in distances:
        velocities.append(calculate_velocity_at_distance(v0, drag_coefficient, bullet_mass, bullet_area, d, angle))
    return velocities


def calculate_time_of_flight(v0, drag_coefficient, bullet_mass, bullet_area, distance, angle):
    """
    Calculate the time of flight to reach a given distance.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distance (float): Distance to the target in meters.
    angle (float): Barrel angle in degrees.

    Returns:
    float: Time of flight in seconds.
    """
    if distance == 0:
        return 0

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction (0 for horizontal shot)
    v0z = 0  # Initial velocity in z-direction (no initial drift)
    y0 = [0, 0, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    # Estimate max flight time using initial horizontal velocity
    t_max = 1.5 * distance / v0x

    # Perform numerical integration using Runge-Kutta method
    sol = solve_ivp(
        bullet_dynamics,
        [0, t_max],
        y0,
        args=(drag_coefficient, bullet_mass, bullet_area),
        method='RK45',  # Use default Runge-Kutta method
        rtol=1e-8,  # Relative tolerance for precision
        atol=1e-10
    )

    # Capture results
    x = sol.y[0]  # Horizontal positions
    y = sol.y[1]  # Vertical positions

    # Find the time when the bullet reaches the target distance
    distances = sol.y[0]
    index = np.argmax(distances >= distance)
    if index == 0:
        raise RuntimeError("Solution did not reach the specified distance.")

    # Interpolate the time at which the bullet reaches the exact distance
    x_before = distances[index - 1]
    x_after = distances[index]
    t_before = sol.t[index - 1]
    t_after = sol.t[index]

    # Linear interpolation for time at the exact distance
    t_target = t_before + (distance - x_before) * (t_after - t_before) / (x_after - x_before)

    return t_target


def calculate_time_of_flights(v0, drag_coefficient_g1, bullet_mass, bullet_area, distances, angle):
    """
    Calculate the time of flights to reach multiple distances.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient_g1 (float): Drag coefficient.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distances (list[float]): List of distances in meters.
    angle (float): Barrel angle in degrees.

    Returns:
    list[float]: List of time of flights to each distance in seconds.
    """
    time_to_distances = []
    for d in distances:
        time_to_distances.append(
            calculate_time_of_flight(v0, drag_coefficient_g1, bullet_mass, bullet_area, d, angle))
    return time_to_distances

def calculate_poi(v0, d_target, drag_coefficient, bullet_mass, bullet_area, hob, angle):
    """
    Calculate the point of impact (POI) at a given target distance using numerical integration.

    Parameters:
    d_target (float): Target distance in meters.
    drag_coefficient (float): Drag coefficient.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    hob (float): Height over bore in meters.
    v0 (float): Initial velocity in m/s.
    angle (float): Barrel angle in radians.

    Returns:
    float: Final vertical position at the target distance in meters.
    """
    if d_target == 0:
        # For zero distance, return the initial height above ground level
        return np.nan

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction (0 for horizontal shot)
    v0z = 0  # Initial velocity in z-direction (no initial drift)

    y0 = [0, hob, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    t_max = d_target / v0x
    # Ensure t_eval is a 1-dimensional array
    t_eval = np.linspace(0, t_max, 1000)

    # Use adaptive integration (Runge-Kutta method)
    # sol = solve_ivp(bullet_dynamics, [0, t_max], y0, args=(drag_coefficient, bullet_mass), t_eval=t_eval)

    sol = solve_ivp(
        bullet_dynamics,
        [0, t_max],
        y0,
        args=(drag_coefficient, bullet_mass, bullet_area),
        t_eval=t_eval,
        method='RK45',  # More stable for complex problems
        rtol=1e-8,  # Relative tolerance for precision
        atol=1e-10
    )

    x = sol.y[0]  # Horizontal positions
    y = sol.y[1]  # Vertical positions

    return y[-1] - hob  # Difference between final y position and initial height


def calculate_pois(v0, drag_coefficient, bullet_mass, bullet_area, hob, angle, distances):
    """
    Calculate points of impact (POIs) for multiple distances using numerical integration.

    Parameters:
    drag_coefficient (float): Drag coefficient.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    hob (float): Height over bore in meters.
    v0 (float): Initial velocity in m/s.
    angle (float): Barrel angle in radians.
    distances (array-like): Array of distances in meters at which to compute POIs.

    Returns:
    np.array: Array of points of impact for each distance.
    """
    pois = []
    for d in distances:
        poi = calculate_poi(v0, d, drag_coefficient, bullet_mass, bullet_area, hob, angle)
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
    k (float): Point of impact in meters.
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

def angle_to_mrads(angle: float, distance: float) -> float:
    return angle / distance * 1000.0

def calculate_coriolis_drifts(v0, drag_coefficient, bullet_mass, bullet_area, distances, latitude):
    """
    Calculate the Coriolis effect using the solve_ivp method over an array of distances, accounting for drag.

    Parameters:
    v0 (float): Initial horizontal velocity of the object (m/s)
    drag_coefficient (float): Drag coefficient of the bullet
    bullet_mass (float): Mass of the bullet (kg)
    bullet_area (float): Cross-sectional area of the bullet (m^2)
    distances (list): Array of distances in meters (cumulative steps)
    latitude (float): Latitude in degrees

    Returns:
    tuple: A tuple containing the total drift and a list of drifts at each step
    """

    def coriolis_ode(t, state, drag_coefficient, bullet_area, bullet_mass, omega, phi):
        """
        Computes derivatives for the Coriolis effect ODE, accounting for drag.

        Parameters:
        state (list): Current state [vx, x_drift]
        v0 (float): Initial horizontal velocity of the object (m/s)
        drag_coefficient (float): Drag coefficient of the bullet
        bullet_area (float): Cross-sectional area of the bullet (m^2)
        bullet_mass (float): Mass of the bullet (kg)
        omega (float): Angular velocity of the Earth in rad/s
        phi (float): Latitude in radians

        Returns:
        list: Derivatives [dvx/dt, dx_drift/dt]
        """
        vx, x_drift = state

        # Calculate drag force and acceleration
        drag = calculate_drag_force(vx, drag_coefficient, bullet_area)
        a_drag = drag / bullet_mass

        # Coriolis effect component in x-direction
        a_coriolis = 2 * vx * omega * np.sin(phi)

        # Derivatives
        dvx_dt = -a_drag
        dx_drift_dt = a_coriolis

        return [dvx_dt, dx_drift_dt]

    omega = 7.2921e-5  # Angular velocity of the Earth in rad/s
    phi = np.radians(latitude)

    # Initial conditions: velocity and drift
    init_state = [v0, 0]

    # Solve the ODE
    sol = solve_ivp(
        coriolis_ode,
        [0, distances[-1] / v0],  # Time span: from 0 to the time it takes to travel the maximum distance
        init_state,
        args=(drag_coefficient, bullet_area, bullet_mass, omega, phi),
        t_eval=[d / v0 for d in distances]  # Evaluation points based on distances
    )

    # Extract results
    drifts = sol.y[1]
    total_drift = drifts[-1]

    return total_drift, drifts


def calculate_spin_drift(v0, drag_coefficient, target_distance, bullet_mass, bullet_area, twist_rate, angle):
    """
    Calculate the spin drift of a bullet at a given target distance.

    Parameters:
    v0 (float): Initial velocity of the bullet in m/s.
    drag_coefficient (float): Drag coefficient of the bullet.
    target_distance (float): Distance to the target in meters.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    twist_rate (float): Twist rate of the barrel in meters.
    param angle (float): Barrel angle in radians

    Returns:
    float: Adjusted spin drift in meters.
    """
    # Calculate velocity at the given distance considering drag
    velocity = calculate_velocity_at_distance(v0, drag_coefficient, bullet_mass, bullet_area, target_distance, angle)

    # Bullet spin rate (radians per second)
    spin_rate = (2 * pi * velocity) / twist_rate

    # Empirical constant for spin drift (this value can be adjusted based on experimental data)
    k = 2.25e-6  # Adjust based on bullet type
    # Simplified estimation of the gyroscopic effect
    spin_drift = (spin_rate * target_distance ** 2) / (v0 ** 2)

    # Multiply by empirical constant
    adjusted_spin_drift = k * spin_drift

    return adjusted_spin_drift


def calculate_spin_drifts(v0, distances, drag_coefficient, bullet_mass, bullet_area, twist_rate, angle):
    drifts = []
    for d in distances:
        drifts.append(calculate_spin_drift(v0, drag_coefficient, d, bullet_mass, bullet_area, twist_rate, angle))
    return drifts

# Windage calculations

def calculate_wind_drift_at_distance(v0, drag_coefficient, bullet_mass, bullet_area, wind_speed, wind_angle, distance, angle):
    if distance == 0:
        return 0

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction (0 for horizontal shot)
    v0z = 0  # Initial velocity in z-direction (no initial drift)
    y0 = [0, 0, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    # Increase the estimated time of flight
    t_max = calculate_time_of_flight(v0, drag_coefficient, bullet_mass, bullet_area, distance, angle) * 2
    t_span = [0, t_max]

    sol = solve_ivp(
        bullet_dynamics,
        t_span,
        y0,
        args=(drag_coefficient, bullet_mass, bullet_area, wind_speed, wind_angle),
        method='LSODA',
        dense_output=True,
        rtol=1e-8,
        atol=1e-10
    )
    x_vals = sol.y[0]  # Horizontal positions
    z_vals = sol.y[2]  # Z-direction positions (wind drift)

    index = np.argmax(x_vals >= distance)

    if index == 0:
        raise RuntimeError("Bullet did not reach the specified distance.")

    x_before = x_vals[index - 1]
    x_after = x_vals[index]
    z_before = z_vals[index - 1]
    z_after = z_vals[index]

    z_target = z_before + (distance - x_before) * (z_after - z_before) / (x_after - x_before)

    return z_target


def calculate_wind_drifts(v0, drag_coefficient, bullet_mass, bullet_area, distances, wind_speed, wind_angle, angle):
    """
    Calculate wind drifts at multiple distances.

    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_mass (float): Mass of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distances (list[float]): List of distances in meters.
    wind_speed (float): Wind speed in m/s.
    wind_angle (float): Wind angle in degrees.
    angle (float): barrel angle in degrees.

    Returns:
    list[float]: List of wind drifts at each distance in meters.
    """
    drifts = []
    for d in distances:
        drifts.append(
            calculate_wind_drift_at_distance(v0, drag_coefficient, bullet_mass, bullet_area, wind_speed, wind_angle, d, angle))
    return drifts

def calculate_mpbr(v0, drag_coefficient, bullet_mass, bullet_area, target_size, hob, d_zero, angle):
    """
    Calculate the Maximum Point Blank Range (MPBR) for a bullet.

    v0: Initial velocity (m/s).
    drag_coefficient: Drag coefficient.
    bullet_mass: Mass of the bullet (kg).
    bullet_area: Cross-sectional area of the bullet (m^2).
    air_density: Air density (kg/m^3).
    target_size: Vertical size of the target (m).
    sight_height: Height of the sight above the bore (m), default is 0.
    zero_distance: Distance at which the rifle is zeroed (m), default is 100 m.

    Returns: The MPBR (m).
    """
    # Define the allowable vertical deviation (half the target size)
    max_rise = target_size / 2  # The maximum allowed rise
    max_fall = target_size / 2  # The maximum allowed fall below the line of sight

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction (0 for horizontal shot)
    v0z = 0  # No initial lateral velocity (z-direction)

    # Initial state for the bullet's motion, considering sight height
    y0 = [0, -hob, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    # Estimate the maximum time of flight based on typical flight times
    t_max = calculate_time_of_flight(v0, drag_coefficient, bullet_mass, bullet_area, d_zero, angle) * 2
    t_span = [0, t_max]

    # Solve the ODE using your bullet_dynamics function
    sol = solve_ivp(
        bullet_dynamics,  # Your dynamics function
        t_span,
        y0,
        args=(drag_coefficient, bullet_mass, bullet_area),
        method='LSODA',
        dense_output=True,
        rtol=1e-8,
        atol=1e-10
    )

    # Extract horizontal positions (x) and vertical positions (y)
    x_vals = sol.y[0]  # Horizontal positions (distance)
    y_vals = sol.y[1]  # Vertical positions (height)

    # Find the maximum distance where the bullet's height stays within the allowed limits
    mpbr = 0  # Initialize MPBR

    for x, y in zip(x_vals, y_vals):
        # Check if the bullet's height is within the acceptable rise and fall range
        if -max_fall <= y <= max_rise:
            mpbr = x  # Update MPBR to the current distance where it's within the target size
        else:
            break  # Once the bullet goes outside the acceptable range, stop

    return mpbr

def calculate_hold_mrad(target_speed, target_distance, target_angle, flight_time):
    """
    Calculate the hold (lead) for a moving target in milliradians (mrad).

    Parameters:
    target_speed (float): Speed of the target in meters per second (m/s).
    target_distance (float): Distance to the target in meters (m).
    target_angle (float): Angle of the target's motion in degrees.
    flight_time (float): Flight time of the projectile in seconds (s).

    Returns:
    float: Required hold (lead) in milliradians (mrad).
    :param target_angle:
    """
    # The hold is the product of the target's speed and the flight time of the projectile

    if target_distance <=0:
        return 0
    hold_m = target_speed * flight_time * math.cos(radians(target_angle))
    # Convert hold to milliradians
    hold_mrad = (hold_m / target_distance) * 1000
    return hold_mrad

def create_hold_table(vt_arr, d_arr, t_arr, target_angle=0):
    """
    Create a table of holds for a range of target speeds and distances.

    Parameters:
    vt_arr (array-like): Array of target speeds in km/hob.
    d_arr (array-like): Array of distances to the target in meters.

    Returns:
    np.array: 2D array of holds in milliradians (mrad).
    """
    # Check that the lengths of d_arr and t_arr are the same
    if len(d_arr) != len(t_arr):
        raise ValueError("Distances, velocities and flight times arrays must be the same length.")

    # Initialize an empty table to store the holds
    hold_table = np.zeros((len(vt_arr), len(d_arr)))
    # Recalculate velocities to m/s
    vt_arr = np.array(vt_arr) * KMH_TO_MPS

    # Calculate the hold for each combination of target speed and distance
    for i, vt in enumerate(vt_arr):
        for j, d in enumerate(d_arr):
            hold_table[j, i] = np.round(calculate_hold_mrad(vt, d, target_angle, t_arr[j]), 1)
    return hold_table

def calculate_projectile_3d_trajectory(v0, drag_coefficient, distance, bullet_weight, bullet_area,
                                       angle, hob, wind_speed=0, wind_angle=0, dt=0.01):
    """
    Simulate 3D bullet trajectory and stop integration exactly at given distance.
    """

    wind_angle = np.radians(wind_angle)
    v0x = v0 * np.cos(angle)
    v0y = v0 * np.sin(angle)
    v0z = 0.0
    y0 = [0.0, hob, 0.0, v0x, v0y, v0z]
    t_max = 1.5 * distance / v0x
    t_eval = np.arange(0, t_max, dt)

    def make_stop_at_distance_event(target_x):
        def event(t, y, *args):
            return y[0] - target_x
        event.terminal = True
        event.direction = 1
        return event

    stop_event = make_stop_at_distance_event(distance)

    sol = solve_ivp(
        fun=bullet_dynamics,
        t_span=(0, t_max),
        y0=y0,
        args=(drag_coefficient, bullet_weight, bullet_area, wind_speed, wind_angle),
        t_eval=t_eval,
        events=stop_event,
        method='LSODA',
        dense_output=True,  # <--- important
        rtol=1e-8,
        atol=1e-10
    )

    if sol.t_events[0].size > 0:
        impact_time = sol.t_events[0][0]
        dense_point = sol.sol(impact_time)  # interpolated state [x, y, z, vx, vy, vz]

        mask = sol.t < impact_time
        x_vals = np.append(sol.y[0][mask], dense_point[0])
        y_vals = np.append(sol.y[1][mask], dense_point[1])
        z_vals = np.append(sol.y[2][mask], dense_point[2])
        t_eval = np.append(sol.t[mask], impact_time)
    else:
        x_vals = sol.y[0]
        y_vals = sol.y[1]
        z_vals = sol.y[2]
        t_eval = sol.t

    return x_vals, y_vals, z_vals, t_eval

