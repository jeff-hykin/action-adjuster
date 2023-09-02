from trivial_torch_tools import Sequential
import torch
import torch.nn as nn

layers = Sequential(input_shape=(1,28,28))
layers.add_module('conv1', nn.Conv2d(1, 10, kernel_size=5))
layers.add_module('conv1_pool', nn.MaxPool2d(2))
layers.add_module('conv1_activation', nn.ReLU())
layers.add_module('conv2', nn.Conv2d(10, 10, kernel_size=5))
layers.add_module('conv2_drop', nn.Dropout2d())
layers.add_module('conv2_pool', nn.MaxPool2d(2))
layers.add_module('conv2_activation', nn.ReLU())
layers.add_module('flatten', nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
layers.add_module('fc1', nn.Linear(layers.output_size, 10))
layers.add_module('fc1_activation', nn.LogSoftmax(dim=1))
