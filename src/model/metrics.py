import tensorflow as tf

def iou_score(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    union = tf.reduce_sum(y_true + y_pred, axis=(1,2)) - intersection
    iou = (intersection + 1e-15) / (union + 1e-15)
    return tf.reduce_mean(iou)

def jaccard_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    union = tf.reduce_sum(y_true + y_pred, axis=(1,2)) - intersection
    jaccard = (intersection + 1e-15) / (union + 1e-15)
    return 1 - tf.reduce_mean(jaccard)

def f1_score(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    precision = intersection / (tf.reduce_sum(y_pred, axis=(1,2)) + 1e-15)
    recall = intersection / (tf.reduce_sum(y_true, axis=(1,2)) + 1e-15)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    return tf.reduce_mean(f1)

def precision(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    precision = intersection / (tf.reduce_sum(y_pred, axis=(1,2)) + 1e-15)
    return tf.reduce_mean(precision)

def recall(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    recall = intersection / (tf.reduce_sum(y_true, axis=(1,2)) + 1e-15)
    return tf.reduce_mean(recall)