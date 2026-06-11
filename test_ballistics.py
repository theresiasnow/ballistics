"""Regression tests for the ballistics library.

Run with: uv run pytest test_ballistics.py
"""

import ballistics as b
import util
from ballistics import Load


def setup_module(_module):
    # Standard-ish atmosphere used across the tests.
    util.set_air_density(b.calculate_air_density(13.0, 1020 * 100, 0.77))


def _zeroed(load):
    angle = b.calibrate_zero(load.v0, load.zero_distance_m, load.bc_metric,
                             load.sight_height_m, load.drag)
    return angle


def test_bc_conversion_includes_pound_factor():
    # 0.243 lb/in^2 -> kg/m^2 must include both lb->kg and in->m^2.
    expected = 0.243 * 0.45359237 / 0.0254 ** 2
    assert abs(b.convert_bc_to_metric(0.243) - expected) < 1e-9


def test_sight_height_changes_trajectory():
    """Regression: sight height must affect the trajectory.

    A previous bug launched the bullet at +hob and subtracted hob on return,
    cancelling sight height out entirely. Two different sight heights must give
    measurably different elevation corrections, especially at short range.
    """
    base = dict(v0=836, bc=0.160, drag_model="G7", bullet_mass_gr=69,
                bullet_diameter_in=0.224, bullet_length_in=1.0, twist_in=7)
    low = Load("low", sight_height_mm=38.1, **base)
    high = Load("high", sight_height_mm=70.0, **base)

    a_low, a_high = _zeroed(low), _zeroed(high)
    e_low = b.poi_to_mrad(b.calculate_poi(low.v0, 50, low.bc_metric, low.sight_height_m, a_low, low.drag), 50)
    e_high = b.poi_to_mrad(b.calculate_poi(high.v0, 50, high.bc_metric, high.sight_height_m, a_high, high.drag), 50)

    # At 50 m the two sight heights must differ by an appreciable amount.
    assert abs(e_low - e_high) > 0.1


def test_zero_is_on_the_line_of_sight():
    """At the zero distance the bullet is on the line of sight (POI ~ 0)."""
    load = Load("GTX_69gr", v0=836, bc=0.160, drag_model="G7", bullet_mass_gr=69,
                bullet_diameter_in=0.224, bullet_length_in=1.0, twist_in=7,
                sight_height_mm=63.5, zero_distance_m=100)
    angle = _zeroed(load)
    poi = b.calculate_poi(load.v0, load.zero_distance_m, load.bc_metric,
                          load.sight_height_m, angle, load.drag)
    assert abs(poi) < 1e-3  # within 1 mm of the line of sight


def test_drop_and_velocity_are_monotonic():
    """Drop increases (more negative) and velocity decreases with distance."""
    load = Load("GTX_69gr", v0=836, bc=0.160, drag_model="G7", bullet_mass_gr=69,
                bullet_diameter_in=0.224, bullet_length_in=1.0, twist_in=7,
                sight_height_mm=63.5, zero_distance_m=100)
    angle = _zeroed(load)
    drops, vels = [], []
    for d in (150, 300, 450, 600):
        drops.append(b.calculate_poi(load.v0, d, load.bc_metric, load.sight_height_m, angle, load.drag))
        vels.append(b.calculate_velocity_at_distance(load.v0, load.bc_metric, d, angle, load.drag))
    assert all(drops[i] > drops[i + 1] for i in range(len(drops) - 1))   # more negative
    assert all(vels[i] > vels[i + 1] for i in range(len(vels) - 1))       # slowing
    assert vels[-1] < load.v0


def test_g7_retains_velocity_better_than_low_bc():
    """A higher-BC G7 bullet retains more velocity downrange than a low-BC one."""
    angle = 0.0
    hi = b.calculate_velocity_at_distance(836, b.convert_bc_to_metric(0.300), 500, angle, b.DragModel("G7"))
    lo = b.calculate_velocity_at_distance(836, b.convert_bc_to_metric(0.100), 500, angle, b.DragModel("G7"))
    assert hi > lo


def test_aerodynamic_jump_sign_and_magnitude():
    """Aero jump flips sign with crosswind direction and is zero in calm air."""
    load = Load("Fiocchi", v0=845, bc=0.160, drag_model="G7", bullet_mass_gr=69,
                bullet_diameter_in=0.224, bullet_length_in=0.905, twist_in=7,
                sight_height_mm=63.5, zero_distance_m=100)
    args = (load.twist_m, load.bullet_length_m, load.bullet_diameter_m, load.bullet_mass_kg)

    assert b.calculate_aerodynamic_jump(*args, 0.0) == 0.0
    left = b.calculate_aerodynamic_jump(*args, -2.0)
    right = b.calculate_aerodynamic_jump(*args, 2.0)
    assert left == -right                       # symmetric in wind direction
    assert 0.05 < abs(left) < 0.5               # right ballpark (~0.1 mrad / m/s)


def test_drop_decreases_with_higher_muzzle_velocity():
    """A faster muzzle velocity gives less drop (more positive POI) at range."""
    base = dict(bc=0.160, drag_model="G7", bullet_mass_gr=69, bullet_diameter_in=0.224,
                bullet_length_in=0.905, twist_in=7, sight_height_mm=63.5, zero_distance_m=100)
    slow = Load("slow", v0=820, **base)
    fast = Load("fast", v0=870, **base)
    a_slow, a_fast = _zeroed(slow), _zeroed(fast)
    d = 500
    drop_slow = b.calculate_poi(slow.v0, d, slow.bc_metric, slow.sight_height_m, a_slow, slow.drag)
    drop_fast = b.calculate_poi(fast.v0, d, fast.bc_metric, fast.sight_height_m, a_fast, fast.drag)
    assert drop_fast > drop_slow  # faster bullet drops less
