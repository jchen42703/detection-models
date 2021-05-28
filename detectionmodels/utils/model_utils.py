def add_metrics(model, metric_dict):
    """
    Adds metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback

    Args:
        model (tf.keras.model): the model to add the metrics to
        metric_dict (dict): Of (metric_name, metric_object)

    Returns:
        null
    """
    for (name, metric) in metric_dict.items():
        model.add_metric(metric, name=name, aggregation='mean')
