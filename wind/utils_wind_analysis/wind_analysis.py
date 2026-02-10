import numpy as np

def get_wind_direction_in_degrees(u, v, grid_angle):
    return (np.degrees(np.arctan2(-u, -v)) - grid_angle + 360) % 360

def get_wind_speed(u, v):
    return (u ** 2 + v ** 2) ** 0.5