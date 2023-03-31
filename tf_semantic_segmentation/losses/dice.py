import tensorflow as tf


def dice_loss():
    def dice_loss(y_true, y_pred):
        """ F1 Score """
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        r = 1 - (numerator + 1) / (denominator + 1)
        return tf.cast(r, tf.float32)

    return dice_loss

def tversky_loss(num_classes=2, beta=0.7, weights=None):
    """ Tversky index (TI) is a generalization of Dice’s coefficient. TI adds a weight to FP (false positives) and FN (false negatives). """
    
    if weights is None:
        weights = [1.0] * num_classes
    weights = tf.cast(tf.convert_to_tensor(weights), tf.float32)
    
    def tversky_loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred * weights)
        false_pos = tf.reduce_sum((1 - y_true) * y_pred * weights)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred) * weights)
        # weighted = [0.001, 1]
        # false_neg = false_neg * [.001, 1]
        # false_pos = false_pos * [.1, 1]
        denominator = numerator + beta * false_pos + (1-beta) * false_neg + 1
        
        r = 1 - (numerator + 1) / denominator
        
        return tf.cast(r, tf.float64)

    return tversky_loss


def focal_tversky_loss(num_classes=2, beta=0.7, gamma=0.75, scale=1, weights=None):
    def focal_tversky(y_true, y_pred):
        loss = tversky_loss(num_classes, beta, weights)(y_true, y_pred)
        return tf.pow(scale*loss, gamma)

    return focal_tversky
