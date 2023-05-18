# 
# image tools
# 
def tensor_from_path(file_path):
    from PIL import Image
    import torchvision.transforms.functional as TF
    image = Image.open(file_path)
    return TF.to_tensor(image)

def pil_image_from_tensor(tensor):
    import torchvision.transforms.functional as TF
    return TF.to_pil_image(tensor)

def torch_tensor_from_opencv_format(array):
    """
    dims in: 1, 210, 160, 3 
    dims out: 1, 3, 210, 160
    """
    tensor = to_tensor(array)
    dimension_count = len(tensor.shape)
    new_shape = [ each for each in range(dimension_count) ]
    height = new_shape[-3]
    width = new_shape[-2]
    channels = new_shape[-1]
    new_shape[-3] = channels
    new_shape[-2] = height
    new_shape[-1] = width
    return tensor.permute(*new_shape)

def opencv_tensor_from_torch_format(array):
    """
    dims in: 1, 3, 210, 160
    dims out: 1, 210, 160, 3
    """
    tensor = to_tensor(array)
    dimension_count = len(tensor.shape)
    new_shape = [ each for each in range(dimension_count) ]
    channels = new_shape[-3]
    height = new_shape[-2]
    width = new_shape[-1]
    new_shape[-3] = height  
    new_shape[-2] = width   
    new_shape[-1] = channels
    return tensor.permute(*new_shape)

def opencv_array_from_pil_image(image):
    import numpy
    return numpy.array(image.convert('RGB') ) 
