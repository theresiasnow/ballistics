"""
Ballistics functions for the ballistics calculator.
input: os.environ['AIR_DENSITY'] need to be set to the air density in kg/m^3
"""
import os
from math import atan, sin, cos, radians, pi

import numpy as np
from scipy.integrate import solve_ivp

# Constants
DEFAULT_AIR_DENSITY = 1.225  # Default air density in kg/m^3 at 15°C, sea level
INCHES_TO_METERS_FACTOR = 0.0254
G = 9.82  # Acceleration due to gravity in m/s^2


def get_air_density() -> float:
    air_density_env = os.getenv('AIR_DENSITY')
    if air_density_env is None:
        print("air density not set using default")
        return DEFAULT_AIR_DENSITY
    return float(air_density_env)


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
    inch_to_meter = INCHES_TO_METERS_FACTOR  # Conversion factor from inches to meters
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


def calculate_drag_force(v, drag_coefficient, bullet_area):
    """
    Calculate the drag force on a bullet.

    Parameters:
    v (float): Velocity of the bullet in m/s.
    drag_coefficient (float): Drag coefficient.
    A (float): Cross-sectional area of the bullet in m^2.

    Returns:
    float: Drag force in N.
    """
    return 0.5 * drag_coefficient * get_air_density() * v ** 2 * bullet_area


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


def calculate_velocity_at_distance(v0, drag_coefficient, bullet_weight, bullet_area, distance, dt=0.01):
    """
    Calculate the velocity at a given distance considering drag using an ODE solver.
    Parameters:
    v0 (float): Initial velocity in m/s.
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    distance (float): Distance travelled in meters.
    dt (float): Time step for the ODE solver.
    Returns:
    float: Velocity at the given distance in m/s.
    """

    def velocity_ode(t, v, drag_coefficient, bullet_weight, bullet_area):
        return -calculate_drag_force(v, drag_coefficient, bullet_area) / bullet_weight

    t_span = (0, distance / v0)  # Time span for integration
    sol = solve_ivp(
        velocity_ode,
        t_span,
        [v0],
        args=(drag_coefficient, bullet_weight, bullet_area),
        method='LSODA',
        max_step=dt)
    return sol.y[0][-1]


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
        time_to_distances.append(
            calculate_time_of_flight(v0, drag_coefficient_g1, bullet_weight, bullet_area, d, angle_rad))
    return time_to_distances


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


def calculate_poi(v0, d_target, drag_coefficient, bullet_weight, bullet_area, hob, angle_rad):
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


def calculate_pois(v0, drag_coefficient, bullet_weight, bullet_area, hob, angle_rad, distances):
    """
    Calculate points of impact (POIs) for multiple distances using numerical integration.

    Parameters:
    drag_coefficient (float): Drag coefficient.
    bullet_weight (float): Weight of the bullet in kg.
    bullet_area (float): Cross-sectional area of the bullet in m^2.
    hob (float): Height over bore in meters.
    v0 (float): Initial velocity in m/s.
    angle_rad (float): Barrel angle in radians.
    distances (array-like): Array of distances in meters at which to compute POIs.

    Returns:
    np.array: Array of points of impact for each distance.
    """
    pois = []
    for d in distances:
        poi = calculate_poi(v0, d, drag_coefficient, bullet_weight, bullet_area, hob, angle_rad)
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


def calculate_coriolis_drift(v0, drag_coefficient, bullet_weight, bullet_area, distances, latitude):
    """
    Calculate the Coriolis effect using the solve_ivp method over an array of distances, accounting for drag.

    Parameters:
    v0 (float): Initial horizontal velocity of the object (m/s)
    drag_coefficient (float): Drag coefficient of the bullet
    bullet_weight (float): Mass of the bullet (kg)
    bullet_area (float): Cross-sectional area of the bullet (m^2)
    distances (list): Array of distances in meters (cumulative steps)
    latitude (float): Latitude in degrees

    Returns:
    tuple: A tuple containing the total drift and a list of drifts at each step
    """

    def coriolis_ode(t, state, v0, drag_coefficient, bullet_area, bullet_weight, omega, phi):
        """
        Computes derivatives for the Coriolis effect ODE, accounting for drag.

        Parameters:
        state (list): Current state [vx, x_drift]
        v0 (float): Initial horizontal velocity of the object (m/s)
        drag_coefficient (float): Drag coefficient of the bullet
        bullet_area (float): Cross-sectional area of the bullet (m^2)
        bullet_weight (float): Weight of the bullet (kg)
        omega (float): Angular velocity of the Earth in rad/s
        phi (float): Latitude in radians

        Returns:
        list: Derivatives [dvx/dt, dx_drift/dt]
        """
        vx, x_drift = state

        # Calculate drag force and acceleration
        drag = calculate_drag_force(vx, drag_coefficient, bullet_area)
        a_drag = drag / bullet_weight

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
        args=(v0, drag_coefficient, bullet_area, bullet_weight, omega, phi),
        t_eval=[d / v0 for d in distances]  # Evaluation points based on distances
    )

    # Extract results
    drifts = sol.y[1]
    total_drift = drifts[-1]

    return total_drift, drifts


def calculate_spin_drift(v0, drag_coefficient, target_distance, bullet_weight, bullet_area, twist_rate):
    """
    Estimate spin drift for a given bullet at a specified range.

    :param target_distance: Range to the target in meters
    :param v0: Muzzle velocity of the bullet in m/s
    :param bullet_length: Length of the bullet in meters
    :param twist_rate: Twist rate of the rifling (in meters per revolution, e.g., 1:7 twist = 0.1778 m/rev)
    :return: Spin drift in meters
    """

    # Calculate velocity at the given distance considering drag
    velocity = calculate_velocity_at_distance(v0, drag_coefficient, bullet_weight, bullet_area, target_distance)

    # Bullet spin rate (radians per second)
    spin_rate = (2 * pi * velocity) / twist_rate

    # Empirical constant for spin drift (this value can be adjusted based on experimental data)
    k = 2.25e-6  # Adjust based on bullet type
    # Simplified estimation of the gyroscopic effect
    spin_drift = (spin_rate * target_distance ** 2) / (v0 ** 2)

    # Multiply by empirical constant
    adjusted_spin_drift = k * spin_drift

    return adjusted_spin_drift


def calculate_spin_drifts(v0, distances, drag_coefficient, bullet_weight, bullet_area, twist_rate):
    drifts = []
    for d in distances:
        drifts.append(calculate_spin_drift(v0, drag_coefficient, d, bullet_weight, bullet_area, twist_rate))
    return drifts

# Windage calculations
def calculate_wind_drift(v0, angle, drag_coefficient, bullet_weight, bullet_area, wind_speed, wind_angle, distances, dt=0.001):
    """
    Compute wind drift (z values) at specified intervals from distances array.
    """

    def dynamics(t, y, drag_coefficient, bullet_weight, bullet_area, wind_speed, wind_angle, air_density):
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

        # Avoid division by zero
        # if relative_velocity == 0:
        #     print("zero division")
        #     relative_velocity = 1e-8

        # Drag force in Newtons (N)
        drag_force = 0.5 * drag_coefficient * bullet_area * air_density * relative_velocity ** 2

        # Accelerations in m/s²
        ax = -(drag_force * relative_vx) / (bullet_weight * relative_velocity)
        ay = - G - (drag_force * relative_vy) / (bullet_weight * relative_velocity)
        az = -(drag_force * relative_vz) / (bullet_weight * relative_velocity)

        return [vx, vy, vz, ax, ay, az]

    # Convert wind angle to radians
    wind_angle = np.radians(wind_angle)
    air_density = get_air_density()

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction
    v0z = 0  # Initial velocity in z-direction (no initial drift)
    y0 = [0, 0, 0, v0x, v0y, v0z]

    # Time span
    t_span = (0, calculate_time_of_flight(v0, drag_coefficient, bullet_weight, bullet_area, max(distances), angle))
    print(f"Time span: {t_span} s")

    # Solve the ODE
    sol = solve_ivp(
        dynamics, t_span, y0,
        args=(drag_coefficient, bullet_weight, bullet_area, wind_speed, wind_angle, air_density),
        max_step=dt, method='LSODA', rtol=1e-8, atol=1e-10
    )

    # Extract x and z values (x for distance, z for wind drift)
    x_vals = sol.y[0]  # x positions
    z_vals = sol.y[2]  # z positions (wind drift)

    # Interpolate drift values for the specified distances
    drift_at_distances = np.interp(distances, x_vals, z_vals)

    return drift_at_distances
