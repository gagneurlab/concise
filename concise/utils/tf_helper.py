# helper functions for tensorflow

# copied from
# https://github.com/tensorflow/models/blob/master/inception/inception/slim/losses.py
import tensorflow as tf

# TODO - use the one from tensorflow directly?

def l1_loss(tensor, weight=1.0, scope=None):
    """Define a L1Loss, useful for regularize, i.e. lasso.
    Args:
      tensor: tensor to regularize.
      weight: scale the loss by this factor.
      scope: Optional scope for op_scope.
    Returns:
      the L1 loss op.
    """
    with tf.name_scope(scope, 'L1Loss', [tensor]):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
        return loss


def huber_loss(tensor, k=1, scope=None):
    """Define a huber loss  https://en.wikipedia.org/wiki/Huber_loss 
      tensor: tensor to regularize.
      k: value of k in the huber loss
      scope: Optional scope for op_scope.

    Huber loss:
    f(x) = if |x| <= k:
              0.5 * x^2
           else:
              k * |x| - 0.5 * k^2

    Returns:
      the L1 loss op.
    """
    # assert k >= 0
    with tf.name_scope(scope, 'L1Loss', [tensor]):
        loss = tf.reduce_mean(tf.select(tf.abs(tensor) < k,
                                        0.5 * tf.square(tensor),
                                        k * tf.abs(tensor) - 0.5 * k ^ 2)
                              )
        return loss
