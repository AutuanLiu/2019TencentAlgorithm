import tensorflow as tf

huber_loss = tf.losses.huber_loss


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, delta=clip_delta))
