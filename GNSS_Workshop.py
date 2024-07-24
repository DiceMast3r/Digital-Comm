import numpy as np

lat_a = float(input("Enter the latitude of A: "))
long_a = float(input("Enter the longitude of A: "))
lat_b = float(input("Enter the latitude of B: "))
long_b = float(input("Enter the longitude of B: "))

R = 6371e3

D = (np.pi / 180) * 2 * R * np.arcsin(np.sqrt(np.sin((float(lat_b) - float(lat_a)) / 2) ** 2 + np.cos(float(lat_a)) * np.cos(float(lat_b)) * np.sin((float(long_b) - float(long_a)) / 2) ** 2)) 

print("Coordinates of A: ({}, {})".format(lat_a, long_a))
print("Coordinates of B: ({}, {})".format(lat_b, long_b))
print("Distance between A and B is: {:.7f} m ".format(D))