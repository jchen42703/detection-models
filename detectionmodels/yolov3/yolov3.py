from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from detectionmodels.utils.model_utils import add_metrics
from .loss import yolo3_loss
from .utils import yolo3_body, custom_tiny_yolo3_body
from .postprocess import batched_yolo3_postprocess


YOLO_CONFIG = {
    'tiny_yolo3_darknet': [custom_tiny_yolo3_body, 20,
                           'weights/yolov3-tiny.h5'],
    'yolo3_darknet': [yolo3_body, 185, 'weights/darknet53.h5'],
}


def get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes,
                    input_tensor=None, input_shape=None):
    """
    """
    # prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    # Tiny YOLOv3 model has 6 anchors and 2 feature layers
    if num_feature_layers == 2:
        if model_type in YOLO_CONFIG:
            model_function = YOLO_CONFIG[model_type][0]
            backbone_len = YOLO_CONFIG[model_type][1]
            weights_path = YOLO_CONFIG[model_type][2]

            if weights_path:
                model_body = model_function(
                    input_tensor, num_anchors // 2, num_classes,
                    weights_path=weights_path)
            else:
                model_body = model_function(
                    input_tensor, num_anchors//2, num_classes)
        else:
            raise ValueError('This model type is not supported now')

    # YOLOv3 model has 9 anchors and 3 feature layers
    elif num_feature_layers == 3:
        if model_type in YOLO_CONFIG:
            model_function = YOLO_CONFIG[model_type][0]
            backbone_len = YOLO_CONFIG[model_type][1]
            weights_path = YOLO_CONFIG[model_type][2]

            if weights_path:
                model_body = model_function(
                    input_tensor, num_anchors // 3, num_classes,
                    weights_path=weights_path)
            else:
                model_body = model_function(
                    input_tensor, num_anchors//3, num_classes)
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('model type mismatch anchors')

    return model_body, backbone_len


def get_yolo3_train_model(
        model_type, anchors, num_classes, weights_path=None, freeze_level=1,
        optimizer=Adam(lr=1e-3, decay=0),
        label_smoothing=0, elim_grid_sense=False, model_pruning=False,
        pruning_end_step=10000):
    '''create the training model, for YOLOv3'''
    # K.clear_session() # get a new session
    num_anchors = len(anchors)
    # YOLOv3 model has 9 anchors and 3 feature layers but
    # Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    # so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    # feature map target value, so its shape should be like:
    # [
    #  (image_height/32, image_width/32, 3, num_classes+5),
    #  (image_height/16, image_width/16, 3, num_classes+5),
    #  (image_height/8, image_width/8, 3, num_classes+5)
    # ]
    y_true = [
        Input(
            shape=(None, None, 3, num_classes + 5),
            name='y_true_{}'.format(layer_idx)) for layer_idx in range(
            num_feature_layers)]

    model_body, backbone_len = get_yolo3_model(
        model_type, num_feature_layers, num_anchors, num_classes)
    print('Create {} {} model with {} anchors and {} classes.'.format(
        'Tiny' if num_feature_layers == 2 else '', model_type, num_anchors,
        num_classes))
    print('model layer number:', len(model_body.layers))

    if weights_path:
        # , skip_mismatch=True)
        model_body.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input
        # layers.
        num = (backbone_len, len(model_body.layers)-3)[freeze_level-1]
        for i in range(num):
            model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(
            num, len(model_body.layers)))
    elif freeze_level == 0:
        # Unfreeze all layers.
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable = True
        print('Unfreeze all of the layers.')

    model_loss, location_loss, confidence_loss, class_loss = Lambda(
        yolo3_loss, name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes,
                   'ignore_thresh': 0.5, 'label_smoothing': label_smoothing,
                   'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    loss_dict = {'location_loss': location_loss,
                 'confidence_loss': confidence_loss, 'class_loss': class_loss}
    add_metrics(model, loss_dict)

    # use custom yolo_loss Lambda layer
    model.compile(optimizer=optimizer, loss={
                  'yolo_loss': lambda y_true, y_pred: y_pred})

    return model


def get_yolo3_inference_model(
        model_type, anchors, num_classes, weights_path=None, input_shape=None,
        confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):
    '''create the inference model, for YOLOv3'''
    # K.clear_session() # get a new session
    num_anchors = len(anchors)
    # YOLOv3 model has 9 anchors and 3 feature layers but
    # Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    # so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')

    model_body, _ = get_yolo3_model(
        model_type, num_feature_layers, num_anchors, num_classes,
        input_shape=input_shape)
    print('Create {} YOLOv3 {} model with {} anchors and {} classes.'.format(
        'Tiny' if num_feature_layers == 2 else '', model_type, num_anchors,
        num_classes))

    if weights_path:
        # , skip_mismatch=True)
        model_body.load_weights(weights_path, by_name=False)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(
        batched_yolo3_postprocess, name='yolo3_postprocess',
        arguments={'anchors': anchors, 'num_classes': num_classes,
                   'confidence': confidence, 'iou_threshold': iou_threshold,
                   'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model
