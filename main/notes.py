import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, impulse
import math
from __dependencies__ import blissful_basics as bb

import do_mpc

# 
# model
# 
if True:
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # x_velocity = model.set_variable(var_type='_x', var_name='x_velocity', shape=(1,1))
    # y_velocity = model.set_variable(var_type='_x', var_name='y_velocity', shape=(1,1))
    x_position = model.set_variable(var_type='_x', var_name='x_position', shape=(1,1))
    y_position = model.set_variable(var_type='_x', var_name='y_position', shape=(1,1))
    # actual_total_velocity = model.set_variable(var_type='_x', var_name='actual_total_velocity', shape=(1,1))
    # actual_absolute_angle = model.set_variable(var_type='_x', var_name='actual_absolute_angle', shape=(1,1))
    # Two states for the desired (set) motor position:
    desired_total_velocity = model.set_variable(var_type='_u', var_name='desired_total_velocity', shape=(1,1))
    desired_absolute_angle = model.set_variable(var_type='_u', var_name='desired_absolute_angle', shape=(1,1))
    # Uncertainity
    additive_velocity = model.set_variable(var_type='parameter', var_name='additive_velocity', shape=(1,1))
    additive_angle    = model.set_variable(var_type='parameter', var_name='additive_angle', shape=(1,1))

    from casadi import cos, sin
    # link var with its deriviative
    model.set_rhs('x_position', (desired_total_velocity+additive_velocity) * cos(desired_absolute_angle+additive_angle))
    model.set_rhs('y_position', (desired_total_velocity+additive_velocity) * sin(desired_absolute_angle+additive_angle))
    # model.set_rhs('actual_absolute_angle', y_velocity)
    model.setup()


    mpc = do_mpc.controller.MPC(model)

    mpc.set_param(
        n_horizon=10,
        t_step=0.25,
        n_robust=1,
        store_full_solution=True,
    )

    # I don't fully understand this part, I believe it is minimizing these functions
    # but I don't think it is the objective function that optimizes the unknowns
    mpc.set_objective(
        mterm=additive_velocity**2 + additive_angle**2,
        lterm=additive_velocity**2 + additive_angle**2,
    )

    mpc.bounds['lower','_u', 'desired_total_velocity'] = 0
    mpc.bounds['upper','_u', 'desired_total_velocity'] = 100
    mpc.bounds['lower','_u', 'desired_absolute_angle'] = -math.radians(90)
    mpc.bounds['upper','_u', 'desired_absolute_angle'] = math.radians(90)
    # mpc.bounds['lower','_p_est', 'additive_velocity'] = -100 # AttributeError: 'MPC' object has no attribute '_p_est_lb' , 'parameter' also give error
    # mpc.bounds['upper','_p_est', 'additive_velocity'] = 50
    # mpc.bounds['lower','_p_est', 'additive_angle'] = -math.pi
    # mpc.bounds['upper','_p_est', 'additive_angle'] = math.pi

    # provide Uncertainity possibilities
    mpc.set_uncertainty_values(
        additive_velocity=numpy.array(tuple(
            bb.linear_steps(
                start=-100,
                end=50,
                quantity=16
            )
        )),
        additive_angle=numpy.array(tuple(
            bb.linear_steps(
                start=-math.radians(90),
                end=math.radians(90),
                quantity=16
            )
        )),
    )

    mpc.setup()

# 
# simulator
# 
if True:
    simulator = do_mpc.simulator.Simulator(model)
    simulator.settings.t_step = 0.25
    p_template = simulator.get_p_template()
    def p_fun(t_now):
        print(f'''t_now = {t_now}''')
        return p_template

    simulator.set_p_fun(p_fun)

    simulator.setup()


x_position = 0
y_position = 0
inital_state = np.array([x_position, y_position,])
simulator.x0 = inital_state
mpc.x0 = inital_state

# mpc.set_inital_guess() # AttributeError: 'MPC' object has no attribute 'set_inital_guess'

velocity = 2
angle = math.radians(30)
new_x, new_y = simulator.make_step(numpy.array([[velocity],[angle],]))


mpc.make_step(numpy.array([[velocity],[angle],]))
# x,y (position)
# velocity
# theta (angle)


# new position after give velocity, and angle

# deriviative(x) = velocity * cos(theta)
# deriviative(y) = velocity * sin(theta)
# deriviative(theta) = ?
# deriviative(velocity) = ?


# input_values = [ (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0) ]
# output_positions = [ (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0) ]

# def predict_function(x_prev, y_prev, velocity, angle, unknown1, unknown2):
#     velocity = velocity + unknown  
#     angle = angle + unknown2
#     return (x_prev + velocity * np.cos(angle), y_prev + velocity * np.sin(angle))



# # # Create an LTI system
# # # example: second-order system
# # xi = 0.5  # Damping ratio
# # omega_n = 2.0  # Natural frequency
# # system = lti([1], [1, 2 * xi * omega_n, omega_n**2])


# # time_values = np.linspace(0, 10, 1000)
# # response = np.zeros_like(time_values)
# # # Compute and accumulate responses to each impulse
# # impulse_times = [1.0, 3.0, 5.0]
# # for impulse_time in impulse_times:
# #     _, h = impulse(system, T=time_values)
# #     response += np.roll(h, int(impulse_time * 1000))
# #     import code; code.interact(banner='',local={**globals(),**locals()})
    
    
# # for time_index in range(200):
    
# #     if time_index == 30:
# #         system = impulse(system, magnitude=1)
    
# #     if time_index == 50:
# #         system = impulse(system, magnitude=0.5)
    
# #     if time_index == 120:
# #         system = impulse(system, magnitude=0.2)
    
    
        

# # # Plot impulse response
# # plt.figure()
# # plt.plot(time_values, response)
# # plt.title('Impulse Response Function')
# # plt.xlabel('Time')
# # plt.ylabel('Response')
# # plt.grid(True)
# # plt.show()
