import tensorflow as tf

@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    """
    Applies a spike function to the given input.

    Args:
        v_scaled (tf.Tensor): The scaled input tensor.
        dampening_factor (tf.Tensor): The dampening factor to be applied.

    Returns:
        tf.Tensor: The output tensor after applying the spike function.

    """
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad



def lif_dynamic(v, i, decay, v_th, dampening_factor=.3):
    """
    Applies a leaky integrate-and-fire (LIF) model to the given input.

    Args:
        v (tf.Tensor): The membrane potential tensor.
        i (tf.Tensor): The input current tensor.
        decay (float): The decay factor.
        v_th (float): The threshold potential.
        dampening_factor (float, optional): The dampening factor. Defaults to .3.

    Returns:
        tuple: The new membrane potential and the new spike function.
    """

    old_z = spike_function((v - v_th) / v_th, dampening_factor)
    new_v = decay * v + i - old_z * v_th
    new_z = spike_function((new_v - v_th) / v_th, dampening_factor)
    return new_v, new_z

