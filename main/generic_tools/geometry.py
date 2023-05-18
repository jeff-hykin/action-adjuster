import math
def get_distance(x1, y1, x2, y2):
    x_diff = x2 - x1
    y_diff = y2 - y1
    return math.sqrt(x_diff * x_diff + y_diff * y_diff)

def get_angle_from_origin(x, y):
    theta = math.atan2(y, x)
    return zero_to_2pi(theta)

def zero_to_2pi(theta):
    while theta < 0:
        theta = math.tau + theta
    while theta > math.tau:
        theta = theta - math.tau
    return theta

def pi_to_pi(theta):
    while theta < -math.pi:
        theta = theta + math.tau
    while theta > math.pi:
        theta = theta - math.tau
    return theta

def abs_angle_difference(radians1, radians2):
    import math
    radians1_positive = radians1 % (2*math.pi)
    radians2_positive = radians2 % (2*math.pi)
    radians1_offset = (radians1+math.pi) % (2*math.pi)
    radians2_offset = (radians2+math.pi) % (2*math.pi)
    
    # 350° - 5°
    # ((350°+180) % 360) - ((5°+180) % 360)
    # 170 - 185 => 15
    
    return min(
        abs(radians1_positive-radians2_positive),
        abs(radians1_offset-radians2_offset),
    )