"""
Ballistics functions for the ballistics calculator.

The simulation is a point-mass model with a Mach-dependent standard drag
function (G1 or G7). The retardation experienced by a bullet is

    a_drag = (pi / 8) * rho * v^2 * Cd_ref(Mach) / BC

where BC is the ballistic coefficient in SI area-density units (kg/m^2) and
Cd_ref(Mach) is the standard reference drag coefficient from `drag_tables`.
The bullet's own mass and diameter cancel out of this expression, so the only
projectile inputs needed for the trajectory are the BC and the drag model.

input: os.environ['AIR_DENSITY'] must be set to the air density in kg/m^3
       (use util.set_air_density()).
"""
import math
from dataclasses import dataclass
from math import atan, cos, radians, pi

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

import util as util
from drag_tables import DragModel, speed_of_sound

# Constants
INCHES_TO_METERS_FACTOR = 0.0254  # m per inch
POUNDS_TO_KG_FACTOR = 0.45359237  # kg per pound
GRAINS_TO_KG_FACTOR = 0.00006479891  # kg per grain
KMH_TO_MPS = 1000 / 3600  # Conversion factor from km/h to m/s
G = 9.82  # Acceleration due to gravity in m/s^2

# Earth rotation rate, used by the Coriolis model.
EARTH_OMEGA = 7.2921159e-5  # rad/s

# Default drag model used when a caller does not supply one. A bullet's BC is
# only meaningful together with the reference drag function it was measured
# against, so this default and the BC must agree (a G1 BC needs the G1 model).
DEFAULT_DRAG_MODEL = DragModel("G1")


@dataclass(frozen=True)
class Load:
    """A named rifle/ammunition/scope combination.

    Fields use the units a shooter reads off a box or data sheet (grains,
    inches, lb/in^2). The SI values the simulation needs are exposed as derived
    properties, so notebook code never has to do unit conversions by hand.
    """

    name: str
    v0: float                       # Muzzle velocity, m/s
    bc: float                       # Ballistic coefficient, lb/in^2
    drag_model: str = "G1"          # Reference drag function: "G1" or "G7"
    bullet_mass_gr: float = 0.0     # Bullet mass, grains
    bullet_diameter_in: float = 0.224  # Bullet diameter, inches
    bullet_length_in: float = 0.0   # Bullet length, inches
    twist_in: float = 0.0           # Barrel twist, inches per turn
    sight_height_mm: float = 70.0   # Sight height over bore, mm
    zero_distance_m: float = 100.0  # Zero distance, m

    @property
    def bc_metric(self) -> float:
        """Ballistic coefficient in SI area-density units (kg/m^2)."""
        return convert_bc_to_metric(self.bc)

    @property
    def drag(self) -> DragModel:
        """The standard drag function (G1/G7) this load's BC was measured against."""
        return DragModel(self.drag_model)

    @property
    def bullet_mass_kg(self) -> float:
        return self.bullet_mass_gr * GRAINS_TO_KG_FACTOR

    @property
    def bullet_diameter_m(self) -> float:
        return self.bullet_diameter_in * INCHES_TO_METERS_FACTOR

    @property
    def bullet_length_m(self) -> float:
        return self.bullet_length_in * INCHES_TO_METERS_FACTOR

    @property
    def twist_m(self) -> float:
        return self.twist_in * INCHES_TO_METERS_FACTOR

    @property
    def sight_height_m(self) -> float:
        return self.sight_height_mm / 1000.0


def _resolve_model(drag_model):
    """Return a usable DragModel, falling back to the G1 default."""
    return drag_model if drag_model is not None else DEFAULT_DRAG_MODEL


def _ref_cd(velocity, drag_model):
    """Reference drag coefficient at the given speed for the drag model.

    The Mach number uses the speed of sound at the current firing air
    temperature (set via util.set_air_temperature), not a fixed 15 °C, so the
    drag-curve lookup is correct for the conditions. This matters downrange:
    a wrong speed of sound shifts every Cd lookup and compounds with range.
    """
    mach = velocity / speed_of_sound(util.get_air_temperature())
    return _resolve_model(drag_model).cd(mach)


def bullet_dynamics(t, y, bc_metric, drag_model=None, wind_speed=0.0, wind_angle=0.0):
    """
    Compute the time derivatives of position and velocity for the bullet.

    Parameters:
    bc_metric (float): Ballistic coefficient in SI area-density units (kg/m^2),
        i.e. the imperial BC converted with `convert_bc_to_metric`.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.
    wind_speed (float): Wind speed in m/s.
    wind_angle (float): Wind direction in radians, relative to the line of fire.

    y: Array of [x, y, z, vx, vy, vz] (positions in m, velocities in m/s).

    Returns: derivatives [vx, vy, vz, ax, ay, az] (m/s and m/s^2).
    """
    x, y_pos, z, vx, vy, vz = y

    # Wind components in m/s (wind direction relative to the line of fire)
    wind_vx = wind_speed * np.cos(wind_angle)
    wind_vz = wind_speed * np.sin(wind_angle)

    # Velocity relative to the air mass (drag acts on airspeed, not groundspeed)
    relative_vx = vx - wind_vx
    relative_vy = vy  # Assuming no vertical wind component
    relative_vz = vz - wind_vz

    # Airspeed magnitude in m/s
    relative_velocity = np.sqrt(relative_vx ** 2 + relative_vy ** 2 + relative_vz ** 2)

    # Retardation coefficient k such that the drag deceleration magnitude is
    # k * v^2, acting opposite the airspeed vector:
    #   a_drag = (pi / 8) * rho * Cd_ref(Mach) / BC * v^2
    rho = util.get_air_density()
    cd_ref = _ref_cd(relative_velocity, drag_model)
    k = (pi / 8.0) * rho * cd_ref / bc_metric

    # Accelerations in m/s^2: drag opposes airspeed, gravity acts on -y.
    ax = -k * relative_velocity * relative_vx
    ay = -G - k * relative_velocity * relative_vy
    az = -k * relative_velocity * relative_vz

    return [vx, vy, vz, ax, ay, az]


def calibrate_zero(v0, d_zero, bc_metric, hob, drag_model=None):
    """
    Calibrate the barrel angle to achieve a zero point at a specified distance.

    Parameters:
    v0 (float): Initial velocity of the bullet in m/s.
    d_zero (float): Distance to the zero point in meters.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    hob (float): Height over bore in meters.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    float: The calibrated barrel angle in radians.
    """
    print("Calibrating zero...")
    angle_guess = 0

    # Find the angle that makes the point of impact zero at the zero distance.
    def find_angle(a):
        # fsolve passes the unknown as a 1-element array.
        return calculate_poi(v0, d_zero, bc_metric, hob, a[0], drag_model)

    result = fsolve(find_angle, angle_guess)
    print("Calibration complete. Barrel angle (rad):", result[0])
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
    Calculate the gravity-effective (horizontal) range for an inclined shot.

    Only the horizontal component of range is acted on by the full drop of
    gravity; for an up- or down-hill shot of slant distance `distance` at
    inclination `angle`, the effective range is distance * cos(angle). This is
    the basis of the "rifleman's rule".

    Parameters:
    distance (float): Slant (line-of-sight) distance in meters.
    angle (float): Inclination angle in degrees (0 = horizontal).
    k (float, optional): Optional additive offset in meters. Default 0.0.

    Returns:
    float: The gravity-effective horizontal range in meters.
    """
    return distance * cos(radians(angle)) + k


def calculate_true_ballistic_ranges(distances, angle, k=0.0):
    """
    Vectorized `calculate_true_ballistic_range` over a list of slant distances.

    Parameters:
    distances (list[float]): Slant distances in meters.
    angle (float): Inclination angle in degrees (0 = horizontal).
    k (float, optional): Optional additive offset in meters. Default 0.0.

    Returns:
    np.array: Array of gravity-effective horizontal ranges in meters.
    """
    return np.array([calculate_true_ballistic_range(d, angle, k) for d in distances])


def convert_bc_to_metric(bc):
    """
    Convert a ballistic coefficient from imperial (lb/in^2) to SI (kg/m^2).

    A ballistic coefficient is a sectional-density-like quantity (mass per unit
    frontal area) and is published in lb/in^2. Converting to kg/m^2 therefore
    requires *both* the pound->kilogram factor and the inch->metre factor:

        BC[kg/m^2] = BC[lb/in^2] * (kg/lb) / (m/in)^2

    The previous implementation omitted the pound->kilogram factor, making the
    metric BC 1/0.4536 ~= 2.2x too large (and the drag correspondingly too low).

    Parameters:
    bc (float): Ballistic coefficient in lb/in^2 (the value printed on a box).

    Returns:
    float: Ballistic coefficient in kg/m^2.
    """
    return bc * POUNDS_TO_KG_FACTOR / INCHES_TO_METERS_FACTOR ** 2


def calculate_barrel_angle(hob, poi, d0):
    """
    Calculate the barrel angle required to hit a target.

    Parameters:
    hob (float): Height over bore in meters.
    poi (float): Point of impact offset in meters (negative if below the target).
    d0 (float): Distance to the target in meters.

    Returns:
    float: Barrel angle in radians.
    """
    return atan((hob + poi) / d0)  # Angle in radians


def calculate_velocity_at_distance(v0, bc_metric, distance, angle, drag_model=None):
    """
    Calculate the velocity at a given distance using numerical integration.

    Parameters:
    v0 (float): Initial velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    distance (float): Distance to the target in meters.
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    float: Velocity at the specified distance in m/s.
    """
    if distance == 0:
        return v0

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction
    v0z = 0  # Initial velocity in z-direction (no initial drift)
    y0 = [0, 0, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    # Estimate time of flight, then integrate well past it.
    t_max = calculate_time_of_flight(v0, bc_metric, distance, angle, drag_model) * 2
    t_span = [0, t_max]

    sol = solve_ivp(
        bullet_dynamics,
        t_span,
        y0,
        args=(bc_metric, drag_model),
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

    def interpolate(before, after):
        return before + (distance - x_before) * (after - before) / (x_after - x_before)

    vx_target = interpolate(vx_before, vx_after)
    vy_target = interpolate(vy_before, vy_after)
    vz_target = interpolate(vz_before, vz_after)

    final_velocity = np.sqrt(vx_target ** 2 + vy_target ** 2 + vz_target ** 2)

    return final_velocity


def calculate_velocities(v0, bc_metric, distances, angle, drag_model=None):
    """
    Calculate velocities at multiple distances.

    Parameters:
    v0 (float): Initial velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    distances (list[float]): List of distances in meters.
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    list[float]: List of velocities at each distance in m/s.
    """
    return [
        calculate_velocity_at_distance(v0, bc_metric, d, angle, drag_model)
        for d in distances
    ]


def calculate_time_of_flight(v0, bc_metric, distance, angle, drag_model=None):
    """
    Calculate the time of flight to reach a given distance.

    Parameters:
    v0 (float): Initial velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    distance (float): Distance to the target in meters.
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    float: Time of flight in seconds.
    """
    if distance == 0:
        return 0

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction
    v0z = 0  # Initial velocity in z-direction (no initial drift)
    y0 = [0, 0, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    # Integrate until the bullet crosses the target distance. A terminal event
    # stops the solver exactly at x == distance, so we never have to guess a
    # safe upper bound on flight time (drag makes a fixed multiple unreliable
    # at long range). A generous time ceiling guards against a non-reaching shot.
    def reached_distance(t, y, *args):
        return y[0] - distance
    reached_distance.terminal = True
    reached_distance.direction = 1

    t_ceiling = 10.0 * distance / v0x  # Far beyond any real time of flight
    sol = solve_ivp(
        bullet_dynamics,
        [0, t_ceiling],
        y0,
        args=(bc_metric, drag_model),
        method='RK45',
        events=reached_distance,
        rtol=1e-8,
        atol=1e-10
    )

    if sol.t_events[0].size == 0:
        raise RuntimeError("Solution did not reach the specified distance.")

    return float(sol.t_events[0][0])


def calculate_time_of_flights(v0, bc_metric, distances, angle, drag_model=None):
    """
    Calculate the time of flight to reach multiple distances.

    Parameters:
    v0 (float): Initial velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    distances (list[float]): List of distances in meters.
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    list[float]: List of times of flight to each distance in seconds.
    """
    return [
        calculate_time_of_flight(v0, bc_metric, d, angle, drag_model)
        for d in distances
    ]


def calculate_poi(v0, d_target, bc_metric, hob, angle, drag_model=None):
    """
    Calculate the point of impact (POI) at a given target distance.

    Parameters:
    v0 (float): Initial velocity in m/s.
    d_target (float): Target distance in meters.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    hob (float): Height over bore in meters.
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    float: Vertical position relative to the line of sight at the target (m).
    """
    if d_target == 0:
        # No meaningful point of impact at zero distance.
        return np.nan

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction
    v0z = 0  # Initial velocity in z-direction (no initial drift)

    # The line of sight is the reference (y = 0). The bore sits `hob` below it
    # (sight height over bore), so the bullet launches from y = -hob. The drop
    # at the target is then the bullet's height read directly against y = 0.
    # (Launching at +hob and subtracting hob would cancel, making sight height
    # a no-op -- which is wrong: sight height must affect the near trajectory.)
    y0 = [0, -hob, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    # Integrate until the bullet actually crosses x = d_target. Using a fixed
    # window of d_target / v0x assumes no slowdown, so with drag the bullet
    # would not reach the target within the window and the drop would be read
    # short -- underestimating it, worse at long range. A terminal event stops
    # the solver exactly at x == d_target.
    def reached_distance(t, y, *args):
        return y[0] - d_target
    reached_distance.terminal = True
    reached_distance.direction = 1

    t_ceiling = 10.0 * d_target / v0x  # Far beyond any real time of flight
    sol = solve_ivp(
        bullet_dynamics,
        [0, t_ceiling],
        y0,
        args=(bc_metric, drag_model),
        method='RK45',
        events=reached_distance,
        rtol=1e-8,
        atol=1e-10
    )

    if sol.t_events[0].size == 0:
        # Bullet never reached the target distance (e.g. it fell to ground).
        return np.nan

    # y at the event time is the height (relative to the line of sight) exactly
    # at x = d_target.
    return float(sol.y_events[0][0][1])


def calculate_pois(v0, bc_metric, hob, angle, distances, drag_model=None):
    """
    Calculate points of impact (POIs) for multiple distances.

    Parameters:
    v0 (float): Initial velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    hob (float): Height over bore in meters.
    angle (float): Barrel angle in radians.
    distances (array-like): Distances in meters at which to compute POIs.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    np.array: Array of points of impact for each distance.
    """
    pois = []
    for d in distances:
        poi = calculate_poi(v0, d, bc_metric, hob, angle, drag_model)
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
    float: Required scope correction in milliradians (mrad).
    """
    if d == 0:
        return 0
    return -(poi / d * 1000.0)


def calculate_mrads(distances, pois):
    """
    Calculate the elevation correction in milliradians (mrad) for many distances.

    Parameters:
    distances (list[float]): List of distances in meters.
    pois (list[float]): List of points of impact in meters.

    Returns:
    list[float]: List of elevation corrections in milliradians (mrad).
    """
    mrads: list[int | float] = []
    for i in range(len(distances)):
        mrads.append(poi_to_mrad(pois[i], distances[i]))
    return mrads


def angle_to_mrads(angle: float, distance: float) -> float:
    return angle / distance * 1000.0


def calculate_coriolis_drifts(v0, bc_metric, distances, latitude, azimuth=90.0, drag_model=None):
    """
    Coriolis deflection of the trajectory, accounting for drag.

    The Coriolis acceleration is 2 * Omega x v, where Omega is the Earth's
    rotation vector. In a local frame with x downrange (horizontal), y up, and
    z to the right of the line of fire, and for a firing azimuth measured
    clockwise from true north, the relevant components are:

        a_horizontal (z, rightward) = 2 * Omega * (vx * sin(lat)
                                                    - vy * cos(lat) * cos(az))
        a_vertical   (y, up)        = 2 * Omega * vx * cos(lat) * sin(az)

    The horizontal term is the classic latitude-dependent azimuthal drift; the
    vertical term raises or lowers the impact depending on firing direction.

    Parameters:
    v0 (float): Muzzle velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    distances (list[float]): Distances in meters at which to report drift.
    latitude (float): Latitude in degrees (positive north).
    azimuth (float): Firing azimuth in degrees clockwise from north. Default 90
        (due east), which maximizes the vertical component and gives the
        canonical horizontal drift.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    tuple: (total_horizontal_drift, horizontal_drifts, vertical_drifts) in meters.
    """
    phi = np.radians(latitude)
    az = np.radians(azimuth)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_az, cos_az = np.sin(az), np.cos(az)

    def coriolis_ode(t, state):
        # State: downrange velocity vx and the accumulated horizontal/vertical drift.
        vx, z_drift, vz, y_drift, vy = state

        # Drag deceleration along the line of fire (point-mass, near-horizontal).
        cd_ref = _ref_cd(vx, drag_model)
        k = (pi / 8.0) * util.get_air_density() * cd_ref / bc_metric
        a_drag = k * vx * vx

        # Coriolis accelerations (small relative to drag; integrated for drift).
        a_z = 2 * EARTH_OMEGA * (vx * sin_phi - vy * cos_phi * cos_az)
        a_y = 2 * EARTH_OMEGA * vx * cos_phi * sin_az

        return [-a_drag, vz, a_z, vy, a_y]

    init_state = [v0, 0.0, 0.0, 0.0, 0.0]

    sol = solve_ivp(
        coriolis_ode,
        [0, distances[-1] / v0],
        init_state,
        t_eval=[d / v0 for d in distances],
        rtol=1e-9,
        atol=1e-12,
    )

    horizontal_drifts = sol.y[1]
    vertical_drifts = sol.y[3]
    total_drift = horizontal_drifts[-1]

    return total_drift, horizontal_drifts, vertical_drifts


def miller_stability_factor(twist_rate, bullet_length, bullet_diameter, bullet_mass):
    """
    Miller gyroscopic stability factor Sg (dimensionless).

        Sg = 30 * m / (t^2 * d^3 * l * (1 + l^2))

    with mass m in grains, twist t in calibers/turn, and diameter d and length l
    in calibers (l = length / diameter). All inputs to this function are SI.

    Parameters:
    twist_rate (float): Barrel twist (length per turn) in meters.
    bullet_length (float): Bullet length in meters.
    bullet_diameter (float): Bullet diameter in meters.
    bullet_mass (float): Bullet mass in kg.

    Returns:
    float: Miller stability factor Sg.
    """
    mass_grains = bullet_mass / GRAINS_TO_KG_FACTOR
    diameter_in = bullet_diameter / INCHES_TO_METERS_FACTOR
    length_cal = bullet_length / bullet_diameter
    twist_cal = twist_rate / bullet_diameter

    return (30.0 * mass_grains) / (
        twist_cal ** 2 * diameter_in ** 3 * length_cal * (1.0 + length_cal ** 2)
    )


def calculate_spin_drift(v0, bc_metric, target_distance, twist_rate, bullet_length,
                         bullet_diameter, bullet_mass, angle, drag_model=None):
    """
    Spin (gyroscopic) drift using the Litz approximation.

    The drift of a right-hand-twist bullet is, to good approximation,

        drift[in] = 1.25 * (Sg + 1.2) * tof^1.83

    where tof is the time of flight in seconds and Sg is the Miller gyroscopic
    stability factor (see `miller_stability_factor`). The result is in meters.

    Parameters:
    v0 (float): Muzzle velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    target_distance (float): Distance to the target in meters.
    twist_rate (float): Barrel twist (length per turn) in meters.
    bullet_length (float): Bullet length in meters.
    bullet_diameter (float): Bullet diameter in meters.
    bullet_mass (float): Bullet mass in kg.
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    float: Spin drift in meters (positive = right, for right-hand twist).
    """
    tof = calculate_time_of_flight(v0, bc_metric, target_distance, angle, drag_model)
    sg = miller_stability_factor(twist_rate, bullet_length, bullet_diameter, bullet_mass)

    # Litz spin-drift approximation (inches), then convert to meters.
    drift_inches = 1.25 * (sg + 1.2) * (tof ** 1.83)
    return drift_inches * INCHES_TO_METERS_FACTOR


def calculate_aerodynamic_jump(twist_rate, bullet_length, bullet_diameter, bullet_mass,
                               crosswind_speed):
    """
    Aerodynamic jump: the vertical deflection a crosswind imparts via the
    bullet's gyroscopic response, expressed as a (range-independent) angle.

    A crosswind makes a spin-stabilized bullet jump vertically. Litz gives the
    angle (independent of range, to first order) as

        AJ[mrad] = -0.01 * Sg * crosswind[mph]

    where Sg is the Miller stability factor and the crosswind is the component
    perpendicular to the line of fire. For a right-hand twist, a left-to-right
    crosswind produces a downward jump (negative); the sign flips for the
    opposite wind. Applied Ballistics includes this term in its elevation, which
    is why a right-twist rifle reads a small vertical offset even at its zero.

    Parameters:
    twist_rate (float): Barrel twist (length per turn) in meters.
    bullet_length (float): Bullet length in meters.
    bullet_diameter (float): Bullet diameter in meters.
    bullet_mass (float): Bullet mass in kg.
    crosswind_speed (float): Crosswind component in m/s (positive left-to-right).

    Returns:
    float: Aerodynamic jump angle in milliradians (added to the elevation).
    """
    sg = miller_stability_factor(twist_rate, bullet_length, bullet_diameter, bullet_mass)
    crosswind_mph = crosswind_speed / 0.44704  # m/s -> mph
    return -0.01 * sg * crosswind_mph


def calculate_spin_drifts(v0, distances, bc_metric, twist_rate, bullet_length,
                          bullet_diameter, bullet_mass, angle, drag_model=None):
    """Spin drift at each of several distances (meters). See `calculate_spin_drift`."""
    return [
        calculate_spin_drift(v0, bc_metric, d, twist_rate, bullet_length,
                             bullet_diameter, bullet_mass, angle, drag_model)
        for d in distances
    ]


# Windage calculations

def calculate_wind_drift_at_distance(v0, bc_metric, wind_speed, wind_angle, distance, angle,
                                     drag_model=None):
    """
    Crosswind drift at a single distance (meters).

    Parameters:
    v0 (float): Muzzle velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    wind_speed (float): Wind speed in m/s.
    wind_angle (float): Wind direction in radians, relative to the line of fire.
    distance (float): Distance to the target in meters.
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.
    """
    if distance == 0:
        return 0

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction
    v0z = 0  # Initial velocity in z-direction (no initial drift)
    y0 = [0, 0, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    # Integrate well past the expected time of flight.
    t_max = calculate_time_of_flight(v0, bc_metric, distance, angle, drag_model) * 2
    t_span = [0, t_max]

    sol = solve_ivp(
        bullet_dynamics,
        t_span,
        y0,
        args=(bc_metric, drag_model, wind_speed, wind_angle),
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


def calculate_wind_drifts(v0, bc_metric, distances, wind_speed, wind_angle, angle,
                          drag_model=None):
    """
    Crosswind drift at multiple distances (meters).

    Parameters:
    v0 (float): Muzzle velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    distances (list[float]): Distances in meters.
    wind_speed (float): Wind speed in m/s.
    wind_angle (float): Wind direction in radians, relative to the line of fire.
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.
    """
    return [
        calculate_wind_drift_at_distance(v0, bc_metric, wind_speed, wind_angle, d, angle, drag_model)
        for d in distances
    ]


def calculate_mpbr(v0, bc_metric, target_size, hob, d_zero, angle, drag_model=None):
    """
    Calculate the Maximum Point Blank Range (MPBR) for a bullet.

    Parameters:
    v0 (float): Muzzle velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    target_size (float): Vertical size of the target in meters.
    hob (float): Height over bore (sight height) in meters.
    d_zero (float): Zero distance in meters (used to bound the integration time).
    angle (float): Barrel angle in radians.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    float: The MPBR in meters.
    """
    max_rise = target_size / 2  # Maximum allowed rise above the line of sight
    max_fall = target_size / 2  # Maximum allowed fall below the line of sight

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction
    v0z = 0  # No initial lateral velocity (z-direction)

    # Start at the bore, sight height below the line of sight.
    y0 = [0, -hob, 0, v0x, v0y, v0z]  # [x0, y0, z0, vx0, vy0, vz0]

    t_max = calculate_time_of_flight(v0, bc_metric, d_zero, angle, drag_model) * 2
    t_span = [0, t_max]

    sol = solve_ivp(
        bullet_dynamics,
        t_span,
        y0,
        args=(bc_metric, drag_model),
        method='LSODA',
        dense_output=True,
        rtol=1e-8,
        atol=1e-10
    )

    x_vals = sol.y[0]  # Horizontal positions (distance)
    y_vals = sol.y[1]  # Vertical positions (height vs line of sight)

    mpbr = 0  # Initialize MPBR

    for x, y in zip(x_vals, y_vals):
        if -max_fall <= y <= max_rise:
            mpbr = x  # Still within the target window
        else:
            break  # Bullet has left the acceptable window

    return mpbr


def calculate_hold_mrad(target_speed, target_distance, target_angle, flight_time):
    """
    Calculate the hold (lead) for a moving target in milliradians (mrad).

    Parameters:
    target_speed (float): Speed of the target in m/s.
    target_distance (float): Distance to the target in meters.
    target_angle (float): Angle of the target's motion in degrees (0 = directly
        crossing the line of sight, 90 = moving along it).
    flight_time (float): Flight time of the projectile in seconds.

    Returns:
    float: Required hold (lead) in milliradians (mrad).
    """
    if target_distance <= 0:
        return 0
    # Only the cross-line-of-sight component of motion needs a lead.
    hold_m = target_speed * flight_time * math.cos(radians(target_angle))
    hold_mrad = (hold_m / target_distance) * 1000
    return hold_mrad


def create_hold_table(vt_arr, d_arr, t_arr, target_angle=0):
    """
    Create a table of holds for a range of target speeds and distances.

    Parameters:
    vt_arr (array-like): Array of target speeds in km/h.
    d_arr (array-like): Array of distances to the target in meters.
    t_arr (array-like): Flight times in seconds, one per distance in d_arr.
    target_angle (float): Angle of target motion in degrees.

    Returns:
    np.array: 2D array of holds in milliradians (mrad), indexed [distance, speed].
    """
    if len(d_arr) != len(t_arr):
        raise ValueError("Distances and flight times arrays must be the same length.")

    hold_table = np.zeros((len(d_arr), len(vt_arr)))
    # Convert target speeds from km/h to m/s.
    vt_arr = np.array(vt_arr) * KMH_TO_MPS

    for i, vt in enumerate(vt_arr):
        for j, d in enumerate(d_arr):
            hold_table[j, i] = np.round(calculate_hold_mrad(vt, d, target_angle, t_arr[j]), 1)
    return hold_table


def calculate_projectile_3d_trajectory(v0, bc_metric, distance, angle, wind_speed=0, wind_angle=0,
                                       dt=0.01, drag_model=None):
    """
    Compute the projectile trajectory in 3D space at discrete time intervals.

    Parameters:
    v0 (float): Muzzle velocity in m/s.
    bc_metric (float): Ballistic coefficient in kg/m^2.
    distance (float): Nominal target distance in meters (bounds the integration).
    angle (float): Barrel angle in radians.
    wind_speed (float): Wind speed in m/s.
    wind_angle (float): Wind direction in degrees, relative to the line of fire.
    dt (float): Output time step in seconds.
    drag_model (DragModel | None): Standard drag function (G1/G7). Defaults to G1.

    Returns:
    x_vals, y_vals, z_vals, t_eval: Position arrays and the time grid.
    """
    # Convert wind angle to radians (the dynamics expect radians).
    wind_angle = np.radians(wind_angle)

    # Initial conditions
    v0x = v0 * np.cos(angle)  # Initial velocity in x-direction
    v0y = v0 * np.sin(angle)  # Initial velocity in y-direction
    v0z = 0  # Initial velocity in z-direction (no initial drift)
    y0 = [0, 0, 0, v0x, v0y, v0z]

    # Time span
    t_max = 1.5 * distance / v0x
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)  # Discrete time intervals

    sol = solve_ivp(
        bullet_dynamics,
        t_span,
        y0,
        t_eval=t_eval,
        args=(bc_metric, drag_model, wind_speed, wind_angle),
        method='LSODA',
        rtol=1e-8,
        atol=1e-10
    )

    x_vals = sol.y[0]
    y_vals = sol.y[1]
    z_vals = sol.y[2]

    return x_vals, y_vals, z_vals, t_eval
