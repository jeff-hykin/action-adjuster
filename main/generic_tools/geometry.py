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

def are_points_collinear(p1, p2, p3):
    if (
        (p1[0] == p2[0] and p1[1] == p2[1])
        or (p1[0] == p3[0] and p1[1] == p3[1])
        or (p2[0] == p3[0] and p2[1] == p3[1])
    ):
        return True
    # Calculate slopes
    slope_1 = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] - p1[0] != 0 else float('inf')
    slope_2 = (p3[1] - p2[1]) / (p3[0] - p2[0]) if p3[0] - p2[0] != 0 else float('inf')
    
    # Compare slopes to check collinearity
    return slope_1 == slope_2

def angle_created_by(*, start, midpoint, end):
    if are_points_collinear(start, midpoint, end):
        return math.pi
    # Calculate distances between points
    a = math.sqrt((midpoint[0] - end[0])**2 + (midpoint[1] - end[1])**2)
    b = math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    c = math.sqrt((start[0] - midpoint[0])**2 + (start[1] - midpoint[1])**2)
    
    return zero_to_2pi(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))