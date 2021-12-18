# Import packages and libraries

import math
import numpy as np
import torch


### FUNCTION DEF ###

##### 
## Convert polar angles of a numpy vector [theta, phi] into Cartesian unit vector
#####
def convert_polar_vector_np(angles):
    y = -1 * math.sin(angles[0]) # first column is theta
    x = -1 * math.cos(angles[0]) * math.sin(angles[1])
    z = -1 * math.cos(angles[0]) * math.cos(angles[1])

    mag_v = math.sqrt(x*x + y*y + z*z)
    x /= mag_v
    y /= mag_v
    z /= mag_v
    return np.array([x, y, z])

##### 
## Convert polar angles of tensor batch [[theta, phi] (nx2)] into Cartesian unit vector tensor
#####
def convert_polar_vector(angles):
    y = -1 * torch.sin(angles[:,0]) # first column is theta
    x = -1 * torch.cos(angles[:,0]) * torch.sin(angles[:,1])
    z = -1 * torch.cos(angles[:,0]) * torch.cos(angles[:,1])

    mag_v = torch.sqrt(x*x + y*y + z*z)
    x /= mag_v
    y /= mag_v
    z /= mag_v
    return x, y, z


##### 
## Convert polar angles of single tensor [theta, phi] into Cartesian unit vector tensor
#####
def convert_polar_vector_single(angles):
    y = -1 * torch.sin(angles[0]) # first column of angles is pitches
    x = -1 * torch.cos(angles[0]) * torch.sin(angles[1])
    z = -1 * torch.cos(angles[0]) * torch.cos(angles[1])

    mag_v = torch.sqrt(x*x + y*y + z*z)
    x /= mag_v
    y /= mag_v
    z /= mag_v
    return torch.tensor([x, y, z])