import math
import torch

def get_distance(x1, y1, x2, y2):
    x_diff = x2 - x1
    y_diff = y2 - y1
    return torch.sqrt(x_diff * x_diff + y_diff * y_diff)

def get_angle_from_origin(x, y):
    theta = torch.atan2(y, x)
    return zero_to_2pi(theta)

def zero_to_2pi(theta):
    if theta < 0:
        theta = 2 * torch.pi + theta
    elif theta > 2 * torch.pi:
        theta = theta - 2 * torch.pi
    return theta

def pi_to_pi(theta):
    if theta < -torch.pi:
        theta = theta + 2 * torch.pi
    elif theta > torch.pi:
        theta = theta - 2 * torch.pi
    return theta

def abs_angle_difference(radians1, radians2):
    radians1_positive = radians1 % (2*torch.pi)
    radians2_positive = radians2 % (2*torch.pi)
    radians1_offset = (radians1+torch.pi) % (2*torch.pi)
    radians2_offset = (radians2+torch.pi) % (2*torch.pi)
    
    # 350° - 5°
    # ((350°+180) % 360) - ((5°+180) % 360)
    # 170 - 185 => 15
    return torch.tensor([
        torch.abs(radians1_positive-radians2_positive),
        torch.abs(radians1_offset-radians2_offset),
    ]).min()