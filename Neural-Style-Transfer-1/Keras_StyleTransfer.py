# cell 0
from __future__ import print_function
import os
from os import path

import numpy as np
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

# cell 1
from keras import backend
from keras.models import Model, load_model, save_model
from keras.utils import to_categorical, plot_model

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

# Main Image
content_image = Image.open('content-uag.png')
content_image = content_image.resize((512, 512))

# Style Image
style_image = Image.open('style_monalisa.jpg')
style_image = style_image.resize((512, 512))

# Python Imaging Library SIZE
style_image.size

# Start processing Content Image
# As input data the image gets converted into an array
content_array = np.asarray(content_image, dtype='float32')
# Insert a new axis at the zero position in the expanded array (image shape)
content_array = np.expand_dims(content_array, axis=0)
# content_array=np.array(content_array).copy()

# We do the same for Style Image ...
style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)
# backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))style_array=np.array(style_array).copy()
# (1,512, 512, 3) (Input, Width, Height, Output_channels )
print(content_array.shape)
print(style_array.shape)  # (1,512, 512, 3)

# cell 7
content_array[:, :, :, 0] -= 103.939
content_array[:, :, :, 1] -= 116.779
content_array[:, :, :, 2] -= 123.68
content_array = content_array[:, :, :, ::-1]

# cell 8
style_array[:, :, :, 0] -= 103.939
style_array[:, :, :, 1] -= 116.779
style_array[:, :, :, 2] -= 123.68
style_array = style_array[:, :, :, ::-1]
style_array.shape

# Image dimensions
height = 512
width = 512
content_image = backend.variable(content_array)
style_image = backend.variable(style_array)
combination_image = backend.placeholder((1, height, width, 3))

# cell 10
input_tensor = backend.concatenate(
    [content_image, style_image, combination_image], axis=0)


# cell 11
model = VGG16(input_tensor=input_tensor,
              weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

FINAL_PATH = path.join(path.abspath(
    ""), "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
print(FINAL_PATH)
if path.exists(FINAL_PATH):
    print("Ya existen los pesos para esta configuración")
    model.load_weights(FINAL_PATH)

# cell 12
content_weight = 0.05
style_weight = 2.5
total_variation_weight = 1.0

# cell 13
layers = dict([(layer.name, layer.output) for layer in model.layers])

# Create SVG
'''
FINAL_GRAPHIC_MODEL = path.join(path.abspath(""), "graphic_model.svg")
if not path.exists(path.dirname(FINAL_GRAPHIC_MODEL)):
            os.makedirs(path.dirname(FINAL_GRAPHIC_MODEL))
plot_model(model,
               to_file=FINAL_GRAPHIC_MODEL,
               show_layer_names=True,
               show_shapes=True,
               rankdir="TB")
print(model.summary())
'''

# Content Representation and Loss
loss = backend.variable(0.)

# Mean Square Error: Of the raw tensor-outputs from the layers.


def content_loss(content, combination):
    return backend.sum(backend.square(content-combination))
# content loss (loss between the content image and the output image)


# Given a chosen content layer l, the content loss is defined as
# the Mean Squared Error between the feature map F of our content
# image C and the feature map P of our generated image Y.
layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * \
    content_loss(content_image_features, combination_features)

# Calculate a matrix comprising of correlated features for the tensors
# output by the style layers


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    # Dot product = Vector Of Features activations of a style-layer
    gram = backend.dot(features, backend.transpose(features))
    return gram

# calculate the Mean Squared Error for the Gram-matrices
# instead of the raw tensor-outputs from the layers.


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    st = backend.sum(backend.square(S - C)) / \
        (4. * (channels ** 2) * (size ** 2))
    return st
# one could minimize the losses in the network such that
# the style loss (loss between the output image style and
#  style of ‘style image’)


# cell 19
feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

# cell 20
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight/len(feature_layers))*sl

# the total variation loss (which ensured pixel wise smoothness)
# were at a minimum.


def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :]-x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


# The total loss can then be written as a weighted sum of
# the both the style and content loss
loss += total_variation_weight * total_variation_loss(combination_image)

# cell 22
grads = backend.gradients(loss, combination_image)

#  the output image generated from such a network,
# resembled the input image and had the stylist attributes of the style image
outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)
f_outputs = backend.function([combination_image], outputs)

# cell 24


def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

# cell 25


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


# cell 26
evaluator = Evaluator()

# cell 27
x = np.random.uniform(0, 255, (1, height, width, 3))-128.0

iterations = 10

# cell 28
import time
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Position of the minimum', x, 'and its value', min_val)
    print('Info dict (info): %s' % str(info))
    print('0 if converged, 1 if too many function evaluations or too many iterations, 2 if stopped for another reason, given in info task')
    print('d[task] %s' % str(info['task']))
    print('Grad at Minimum: %s' % str(info['grad']))
    print('Function Calls Made: %s' % str(info['funcalls']))
    print('Number Iterations: %s' % str(info['nit']))
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

# cell 29
print
x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')


# The best results are achieved by a combination of many different
#  layers from the network, which capture both the finer textures
#  and the larger elements of the original image.

# cell 30
resultImage = Image.fromarray(x)
# result = Image.fromarray((visual * 255).astype(numpy.uint8))
resultImage.save('content-uag_monalisa.png')
# resultImage.save('output.bmp')
