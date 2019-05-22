import numpy as np
import tensorflow as tf
from tensorlayer.layers import InputLayer, SubpixelConv2d
from keras.layers import Lambda


def SubpixelConv2D(scale=2,name="subpixel"):
        """
        Keras layer to do subpixel convolution.
        NOTE: Tensorflow backend only. Uses tf.depth_to_space
        
        :param scale: upsampling scale compared to input_shape. Default=2
	:name: name of layer
        :return:
        """

        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape

        '''def subpixel(x):
            return tf.depth_to_space(x, scale)'''
        
        def _phase_shift(x):
            n = InputLayer(x, name='input_subpixel')
            n = SubpixelConv2d(n, scale=scale, n_out_channel=None, act=tf.nn.relu)
            return n.outputs

        return Lambda(_phase_shift, output_shape=subpixel_shape, name=name)
