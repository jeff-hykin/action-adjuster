import numpy
def shift_towards(*, new_value, old_value, proportion):
    if proportion == 1:
        return new_value
    
    values = []
    for old, new in zip(old_value.flat, new_value.flat):
        difference = new - old
        amount = difference * proportion
        values.append(old+amount)
    
    new_array = numpy.array(values)
    return new_array.reshape(old_value.shape)