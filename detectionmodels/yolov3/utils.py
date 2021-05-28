from tensorflow.keras.layers import Add, ZeroPadding2D, UpSampling2D, \
    Concatenate, MaxPooling2D
from tensorflow.keras.models import Model

from detectionmodels.layers.yolo import compose, DarknetConv2D, \
    DarknetConv2D_BN_Leaky, Spp_Conv2D_BN_Leaky


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet53_body(x):
    '''Darknet53 body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(
        x, num_filters, out_filters, predict_filters=None, predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(DarknetConv2D_BN_Leaky(predict_filters, (3, 3)), DarknetConv2D(
        out_filters, (1, 1), name='predict_conv_' + predict_id))(x)
    return x, y


def make_spp_last_layers(
        x, num_filters, out_filters, predict_filters=None, predict_id='1'):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
        DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    if predict_filters is None:
        predict_filters = num_filters*2
    y = compose(DarknetConv2D_BN_Leaky(predict_filters, (3, 3)), DarknetConv2D(
        out_filters, (1, 1), name='predict_conv_' + predict_id))(x)
    return x, y


def yolo3_predictions(
        feature_maps, feature_channel_nums, num_anchors, num_classes,
        use_spp=False):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    # feature map 1 head & output (13x13 for 416 input)
    if use_spp:
        x, y1 = make_spp_last_layers(
            f1, f1_channel_num // 2, num_anchors * (num_classes + 5),
            predict_id='1')
    else:
        x, y1 = make_last_layers(
            f1, f1_channel_num // 2, num_anchors * (num_classes + 5),
            predict_id='1')

    # upsample fpn merge for feature map 1 & 2
    x = compose(
        DarknetConv2D_BN_Leaky(f2_channel_num//2, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, f2])

    # feature map 2 head & output (26x26 for 416 input)
    x, y2 = make_last_layers(
        x, f2_channel_num // 2, num_anchors * (num_classes + 5),
        predict_id='2')

    # upsample fpn merge for feature map 2 & 3
    x = compose(
        DarknetConv2D_BN_Leaky(f3_channel_num//2, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, f3])

    # feature map 3 head & output (52x52 for 416 input)
    x, y3 = make_last_layers(
        x, f3_channel_num // 2, num_anchors * (num_classes + 5),
        predict_id='3')

    return y1, y2, y3


def yolo3_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet53_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    # f1: 13 x 13 x 1024
    f1 = darknet.output
    # f2: 26 x 26 x 512
    f2 = darknet.layers[152].output
    # f3: 52 x 52 x 256
    f3 = darknet.layers[92].output

    f1_channel_num = 1024
    f2_channel_num = 512
    f3_channel_num = 256

    y1, y2, y3 = yolo3_predictions(
        (f1, f2, f3),
        (f1_channel_num, f2_channel_num, f3_channel_num),
        num_anchors, num_classes)

    return Model(inputs, [y1, y2, y3])


def tiny_yolo3_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    # feature map 2 (26x26x256 for 416 input)
    f2 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)

    # feature map 1 (13x13x1024 for 416 input)
    f1 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)))(f2)

    # feature map 1 transform
    x1 = DarknetConv2D_BN_Leaky(256, (1, 1))(f1)

    # feature map 1 output (13x13 for 416 input)
    y1 = compose(DarknetConv2D_BN_Leaky(512, (3, 3)), DarknetConv2D(
        num_anchors*(num_classes+5), (1, 1), name='predict_conv_1'))(x1)

    # upsample fpn merge for feature map 1 & 2
    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x1)

    # feature map 2 output (26x26 for 416 input)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(
            num_anchors * (num_classes + 5),
            (1, 1),
            name='predict_conv_2'))(
        [x2, f2])

    return Model(inputs, [y1, y2])


def custom_tiny_yolo3_body(inputs, num_anchors, num_classes, weights_path):
    '''Create a custom Tiny YOLO_v3 model, use
       pre-trained weights from darknet and fit
       for our target classes.'''
    # TODO: get darknet class number from class file
    num_classes_coco = 80
    base_model = tiny_yolo3_body(inputs, num_anchors, num_classes_coco)
    base_model.load_weights(weights_path, by_name=False)
    print('Load weights {}.'.format(weights_path))

    # get conv output in original network
    y1 = base_model.layers[40].output
    y2 = base_model.layers[41].output
    y1 = DarknetConv2D(
        num_anchors * (num_classes + 5),
        (1, 1),
        name='predict_conv_1')(y1)
    y2 = DarknetConv2D(
        num_anchors * (num_classes + 5),
        (1, 1),
        name='predict_conv_2')(y2)
    return Model(inputs, [y1, y2])
