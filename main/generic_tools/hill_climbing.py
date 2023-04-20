import __dependencies__.blissful_basics as bb
import cma
import numpy

def guess_to_maximize(objective_function, initial_guess, stdev):
    is_scalar = not bb.is_iterable(initial_guess)
    new_objective = objective_function
    if is_scalar: # wrap it
        new_objective = lambda arg1: objective_function(arg1[0])
        initial_guess = [initial_guess, 0]
    else: # flatten it
        initial_guess = numpy.array(initial_guess)
        shape = initial_guess.shape
        if shape == (1,):
            new_objective = lambda arg1: objective_function(numpy.array([arg1[0]]))
        elif len(shape) > 1:
            new_objective = lambda arg1: objective_function(arg1.reshape(shape))
    
    import sys
    
    
    xopt, es = cma.fmin2(
        lambda *args: -new_objective(*args),
        numpy.array(initial_guess.flat),
        stdev,
        options=dict(
            verb_log=0                      ,
            verbose=0                       ,
            verb_plot=0                     ,
            verb_disp=0                     ,
            verb_filenameprefix="/dev/null" ,
            verb_append=0                   ,
            verb_time=False                 ,
        ),
    )
    
    output = xopt
    if is_scalar: # wrap it
        return output[0]
    else: # un-flatten it
        if shape == (1,):
            return numpy.array([output[0]])
        elif len(shape) > 1:
            return output.reshape(shape)
    
    return output