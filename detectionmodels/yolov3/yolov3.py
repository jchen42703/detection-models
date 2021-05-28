from tf.keras.layers import Input
from detectionmodels.utils.model_utils import add_metrics
from .loss import yolo3_loss
from .postprocess import


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
        if model_type in yolo3_tiny_model_map:
            model_function = yolo3_tiny_model_map[model_type][0]
            backbone_len = yolo3_tiny_model_map[model_type][1]
            weights_path = yolo3_tiny_model_map[model_type][2]

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
        if model_type in yolo3_model_map:
            model_function = yolo3_model_map[model_type][0]
            backbone_len = yolo3_model_map[model_type][1]
            weights_path = yolo3_model_map[model_type][2]

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
            name='y_true_{}'.format(l)) for l in range(
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
