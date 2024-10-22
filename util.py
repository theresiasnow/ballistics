# Utils
import json
import os

DEFAULT_AIR_DENSITY = 1.225  # Default air density in kg/m^3 at 15°C, sea level

def set_air_density(air_density: float):
    os.environ['AIR_DENSITY'] = str(air_density)  # kg/m3 needed in ballistics.py

def get_air_density() -> float:
    """
    Get the air density from the environment variable AIR_DENSITY.
    :return: Air density in kg/m^3
    """
    air_density_env = os.getenv('AIR_DENSITY')
    if air_density_env is None:
        print("air density not set using default")
        return DEFAULT_AIR_DENSITY
    return float(air_density_env)

def store_parameters(air_density, barrel_angle, v0, drag_coefficient, h, d_zero, bullet_mass, bullet_area, temp,
                     pressure, humidity):

    parameters = {
        'air_density': air_density,
        'barrel_angle': barrel_angle,
        'v0': v0,
        'drag_coefficient': drag_coefficient,
        'h': h,
        'd_zero': d_zero,
        'bullet_mass': bullet_mass,
        'bullet_area': bullet_area,
        'temp': temp,
        'pressure': pressure,
        'humidity': humidity
    }
    with open('parameters.json', 'w') as f:
        json.dump(parameters, f, indent=4)
    print("Wrote parameters")

def load_parameters():
    try:
        with open('parameters.json', 'r') as f:
            parameters = json.load(f)
    except:
        print("No parameters file found - run main notebook to generate")
    print("Loaded parameters")
    return parameters
