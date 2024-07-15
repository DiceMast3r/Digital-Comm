from sgp4.api import Satrec, jday
from sgp4.api import accelerated
import numpy as np

# Constants
a = 6378.137  # Earth's equatorial radius in km
f = 1 / 298.257223563  # Flattening factor
e2 = f * (2 - f)  # Square of eccentricity

def eci_to_ecef(r, jd, fr):
    """
    Convert ECI coordinates to ECEF coordinates.
    """
    # Calculate the Greenwich Sidereal Time (GST)
    t = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000.0
    GMST = 280.46061837 + 360.98564736629 * (jd + fr - 2451545.0) + 0.000387933 * t**2 - (t**3) / 38710000.0 
    GMST = GMST % 360.0 

    theta = np.radians(GMST)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotation matrix from ECI to ECEF
    R = np.array([
        [ cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0], 
        [        0,        0, 1]
    ]) 

    r_ecef = np.dot(R, r)
    return r_ecef

def ecef_to_latlon(r_ecef):
    """
    Convert ECEF coordinates to geodetic latitude, longitude, and altitude.
    """
    x, y, z = r_ecef
    lon = np.arctan2(y, x)

    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    lat_prev = 0

    # Iteratively compute latitude
    while abs(lat - lat_prev) > 1e-10:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    lat = np.degrees(lat)
    lon = np.degrees(lon)
    return lat, lon, alt

# Example TLE data
line0 = "GPS BIIF-9  (PRN 26)"
line1 = "1 40534U 15013A   24194.06691307  .00000033  00000+0  00000+0 0  9992"
line2 = "2 40534  53.3359 233.3662 0088230  29.7523 330.8139  2.00564147 68101"

# Initialize satellite object
satellite = Satrec.twoline2rv(line1, line2)

# Define the date and time for the calculation (e.g., Julian Date)
year = 2024
month = 7
day = 12
hour = 14
minute = 15
second = 0
jd, fr = jday(year, month, day, hour, minute, second)

# Calculate position and velocity
e, r, v = satellite.sgp4(jd, fr)

if e == 0:
    # Convert ECI to ECEF
    r_ecef = eci_to_ecef(r, jd, fr)

    # Convert ECEF to latitude, longitude, and altitude
    lat, lon, alt = ecef_to_latlon(r_ecef)
    
    print("Satellite: ", line0)
    print("At time (UTC): ", year, month, day, hour, minute, second)
    print("Raw postion vector: ", r)

    print(f"Latitude: {lat:.6f}°")
    print(f"Longitude: {lon:.6f}°")
    print(f"Altitude: {alt:.2f} km")
    print(accelerated)
else:
    print("Error with SGP4 propagation")
