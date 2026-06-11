# Utils
import json
import os
from dataclasses import dataclass

DEFAULT_AIR_DENSITY = 1.225  # Default air density in kg/m^3 at 15°C, sea level
DEFAULT_AIR_TEMPERATURE = 15.0  # Default air temperature in °C (ICAO standard)
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


def set_air_temperature(temperature_celsius: float):
    """Store the firing air temperature (°C) used to compute the speed of sound."""
    os.environ['AIR_TEMPERATURE'] = str(temperature_celsius)


def get_air_temperature() -> float:
    """Air temperature in °C used for the Mach (speed-of-sound) calculation."""
    temp_env = os.getenv('AIR_TEMPERATURE')
    if temp_env is None:
        return DEFAULT_AIR_TEMPERATURE
    return float(temp_env)

@dataclass
class Parameters:
    air_density: float
    barrel_angle: float
    initial_velocity: float
    load_name: str  # Name of the selected Load (see ballistics.Load)
    bc_metric: float  # Ballistic coefficient in kg/m^2 (see ballistics.convert_bc_to_metric)
    drag_model: str  # Standard drag function name, "G1" or "G7"
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
