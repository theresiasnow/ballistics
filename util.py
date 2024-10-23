# Utils
import json
import os
from dataclasses import dataclass

DEFAULT_AIR_DENSITY = 1.225  # Default air density in kg/m^3 at 15°C, sea level
PARAMETERS_FILE_PATH = 'parameters.json'  # Extracted constant


def set_air_density(air_density: float):
    os.environ['AIR_DENSITY'] = str(air_density)  # kg/m3 needed in ballistics.py


def get_air_density() -> float:
    """
    Get the air density from the environment variable AIR_DENSITY.
    :return: Air density in kg/m^3
    """
    air_density_env = os.getenv('AIR_DENSITY')
    if air_density_env is None:
        print("Air density not set, using default.")
        return DEFAULT_AIR_DENSITY
    return float(air_density_env)

@dataclass
class Parameters:
    air_density: float
    barrel_angle: float
    initial_velocity: float
    drag_coefficient: float
    height: float
    zero_distance: float
    bullet_mass: float
    bullet_cross_sectional_area: float
    temperature: float
    pressure: float
    humidity: float


def write_parameters_to_file(parameters: Parameters, filename: str = 'parameters.json'):
    with open(filename, 'w') as f:
        json.dump(parameters.__dict__, f, indent=4)
    print("Wrote parameters to", filename)


def store_parameters(params: Parameters):
    write_parameters_to_file(params)


def read_parameters_from_file() -> Parameters:
    try:
        with open(PARAMETERS_FILE_PATH, 'r') as file:
            parameters_dict = json.load(file)
        print("Loaded parameters")  # Moved inside try block
        return Parameters(**parameters_dict)
    except FileNotFoundError:
        print("No parameters file found - run main notebook to generate")
    except json.JSONDecodeError:
        print("Error decoding JSON from the parameters file")
    return None  # Explicitly return None if there is an error
