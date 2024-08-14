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
line0 = "GPS BIIF-3  (PRN 24)"
line1 = "1 38833U 12053A   24225.24064600  .00000039  00000+0  00000+0 0  9993"
line2 = "2 38833  53.5193 171.1798 0154929  56.6227 304.8345  2.00559544 85920"

# Initialize satellite object
satellite = Satrec.twoline2rv(line1, line2)

# Define the date and time for the calculation (e.g., Julian Date)
year = 2024
month = 8
day = 13
hour_local = 12
minute = 55
second = 0
hour = hour_local - 7

jd, fr = jday(year, month, day, hour, minute, second)


# Calculate position and velocity
e, r, v = satellite.sgp4(jd, fr)

if e == 0:
    # Convert ECI to ECEF
    r_ecef = eci_to_ecef(r, jd, fr)

    # Convert ECEF to latitude, longitude, and altitude
    lat, lon, alt = ecef_to_latlon(r_ecef)
    
    print("Satellite: ", line0)
    print("Date: ", year, month, day)
    print("At time (UTC): {0} hours, {1} minutes, {2} seconds".format(hour, minute, second))
    print("Local time (UTC+7): {0} hours, {1} minutes, {2} seconds".format(hour_local, minute, second))
    print("Raw postion vector: ", r)

    print(f"Latitude: {lat:.6f}°")
    print(f"Longitude: {lon:.6f}°")
    print(f"Altitude: {alt:.2f} km")
    print("C++ accel: " + str(accelerated))
else:
    print("Error with SGP4 propagation")
