import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
from casadi import cos, sin
import do_mpc

def linear_steps(*, start, end, quantity):
    """
        Example:
            assert [4, 11, 18, 24, 31] == list(linear_steps(start=4, end=31, quantity=5))
    """
    import math
    assert quantity > -1
    if quantity != 0:
        quantity = math.ceil(quantity)
        if start == end:
            for each in range(quantity):
                yield start
        else:
            x0 = 1
            x1 = quantity
            y0 = start
            y1 = end
            interpolater = lambda x: y0 if (x1 - x0) == 0 else y0 + (y1 - y0) / (x1 - x0) * (x - x0)
            for x in range(quantity-1):
                yield interpolater(x+1)
            yield end

# 
# model
# 
if True:
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    x_position = model.set_variable(var_type='_x', var_name='x_position', shape=(1,1))
    y_position = model.set_variable(var_type='_x', var_name='y_position', shape=(1,1))
    x_measured = model.set_meas('x_measured', x_position, meas_noise=True)
    y_measured = model.set_meas('y_measured', y_position, meas_noise=True)
    # Two states for the desired (set) motor position:
    desired_total_velocity = model.set_variable(var_type='_u', var_name='desired_total_velocity', shape=(1,1))
    desired_absolute_angle = model.set_variable(var_type='_u', var_name='desired_absolute_angle', shape=(1,1))
    # Uncertainity
    additive_velocity = model.set_variable(var_type='parameter', var_name='additive_velocity', shape=(1,1))
    additive_angle    = model.set_variable(var_type='parameter', var_name='additive_angle', shape=(1,1))
    
    
    # link var with its deriviative
    model.set_rhs('x_position', (desired_total_velocity+additive_velocity) * cos(desired_absolute_angle+additive_angle))
    model.set_rhs('y_position', (desired_total_velocity+additive_velocity) * sin(desired_absolute_angle+additive_angle))
    # model.set_rhs('actual_absolute_angle', y_velocity)
    model.setup()


    mpc = do_mpc.estimator.MHE(model, ['additive_velocity', 'additive_angle'])
    mpc.settings.supress_ipopt_output()

    mpc.set_param(
        t_step=0.25,
        n_horizon=10,
        # n_robust=1,
        store_full_solution=True,
        meas_from_data=True,
    )
    
    
    # I don't understand this part at all
    number_of_inputs = len(model.u.labels())
    number_of_parameters = len(model.p.labels())
    number_of_variables = len(model.x.labels())
    number_of_measured = len(model._y.labels())
    P_v = np.diag(np.array([1]*number_of_measured))
    P_x = np.eye(number_of_variables)
    P_p = 1*np.eye(number_of_inputs)
    # P_w = ???
    
    # FWIW, chatGPT says:
        # P_v (Measurement error weighting matrix): This matrix penalizes the difference between predicted outputs and measured outputs. It is diagonal and contains weights for each measured output. Larger values in this matrix indicate higher confidence in the measurements for corresponding outputs.
        # P_x (State error weighting matrix): This matrix penalizes the difference between predicted states and estimated states. Similar to P_v, it is diagonal and contains weights for each state variable. Larger values in this matrix indicate higher confidence in the initial state estimate for corresponding states.
        # P_p (Input error weighting matrix): This matrix penalizes deviations of the predicted inputs from the actual inputs. It is also diagonal and contains weights for each input variable. Larger values in this matrix indicate higher confidence in the inputs for corresponding input variables.
        # P_w (State regularization weighting matrix): This matrix penalizes changes in states from one time step to the next. It encourages smoothness in state trajectories and helps to avoid overfitting noisy measurements. It is usually a lower triangular matrix with non-negative values on the diagonal and zeros above the diagonal.
    
    mpc.set_default_objective(
        P_v,
        P_x,
        P_p,
        # P_w,
    )
    
    mpc_p_template = mpc.get_p_template()
    @mpc.set_p_fun
    def p_fun_mhe(t_now):
        print(f'''t_now = {t_now}''')
        return mpc_p_template

    mpc.bounds['lower','_u', 'desired_total_velocity'] = 0
    mpc.bounds['upper','_u', 'desired_total_velocity'] = 100
    mpc.bounds['lower','_u', 'desired_absolute_angle'] = -math.radians(90)
    mpc.bounds['upper','_u', 'desired_absolute_angle'] = math.radians(90)
    mpc.bounds['lower','_p_est', 'additive_velocity'] = -100 # AttributeError: 'MPC' object has no attribute '_p_est_lb' , 'parameter' also give error
    mpc.bounds['upper','_p_est', 'additive_velocity'] = 50
    mpc.bounds['lower','_p_est', 'additive_angle'] = -math.pi
    mpc.bounds['upper','_p_est', 'additive_angle'] = math.pi

    # provide Uncertainity possibilities
        # mpc.set_uncertainty_values(
        #     additive_velocity=numpy.array(tuple(
        #         linear_steps(
        #             start=-100,
        #             end=50,
        #             quantity=6,
        #         )
        #     )),
        #     additive_angle=numpy.array(tuple(
        #         linear_steps(
        #             start=-math.radians(90),
        #             end=math.radians(90),
        #             quantity=6,
        #         )
        #     )),
        # )

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
        # value = np.random.uniform(-100,50)
        # print(f'''value = {value}''')
        # p_template['additive_velocity'] = value
        return p_template

    simulator.set_p_fun(p_fun)

    simulator.setup()


initial_x_position = 0
initial_y_position = 0
initial_velocity = 2
initial_angle = math.radians(90)
initial_additive_velocity = 0
initial_additive_angle = 0
inital_state = np.array([initial_x_position, initial_y_position,])
simulator.x0 = inital_state
mpc.x0 = inital_state
mpc.p_est0 = numpy.array([initial_additive_velocity,initial_additive_angle])
mpc.set_initial_guess()


# 
# start simulation
# 

mpc.reset_history()
simulator.reset_history()
number_of_timesteps = 50
actions = [ (1,0), ] * number_of_timesteps
positions = [ ([2],[0]) ] * number_of_timesteps
for each_next_action, each_current_position in zip(actions, positions):
    y0 = simulator.make_step(numpy.array(each_next_action).reshape(2,1), v0=numpy.array([0,0]).reshape(2,1))
    x0 = mpc.make_step(numpy.array(each_current_position)) # MPC estimation step
    print(f'''x0 = {x0}''')

new_x, new_y = simulator.make_step(numpy.array([[initial_velocity],[initial_angle],]))

# 
# simulate data
# 
# simulate not moving
# output = positions[0]

# output = np.array(output)
# output[0] += 1
# output = simulator.make_step(
#     numpy.array([
#         [actions[0][0]],
#         [actions[0][1]],
#     ]),
#     v0=output,
# )

simulator.data['_p','additive_velocity']
simulator.data['_p','additive_angle']
mpc.data['_p','additive_velocity']
mpc.data['_p','additive_angle']