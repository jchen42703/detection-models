"""
Basic wrappers around notable layers with L2 Regularization.
"""
from functools import wraps

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.regularizers import l2

L2_FACTOR = 1e-5


@wraps(Conv2D)
def YoloConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for Conv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    return Conv2D(*args, **yolo_conv_kwargs)


@wraps(DepthwiseConv2D)
def YoloDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for DepthwiseConv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    return DepthwiseConv2D(*args, **yolo_conv_kwargs)
