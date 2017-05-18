import keras.backend as K
import keras.constraints as kc


# TODO - write unit-tests for this unit
#
# TODO - move to a new branch?


class PWMNorm(kc.MinMaxNorm):
    """PWM constraint.
    $$ sum_j w_{ij} = 1 \forall j $$
    $$ w_{ij} \in [0,1] $$
    """

    def __init__(self):
        # TODO - think about the axis
        # MinMaxNorm
        super(PWMNorm, self).__init__(min_value=0.0, max_value=1.0, rate=1.0, axis=0)

    def __call__(self, w):
        # MinMaxNorm
        w = super(PWMNorm, self).__call__(w)

        # TODO - think what is the best way to project to the right manifold

        # TODO - think about the axis

        # unit-norm
        # TODO - prove that this indeed preserves the norm....
        w_unit = w / (K.epsilon() + K.sum(w, axis=0, keepdims=True))
        # axis: integer, axis along which to calculate weight norms.
        #   For instance, in a `Dense` layer the weight matrix
        #   has shape `(input_dim, output_dim)`,
        #   set `axis` to `0` to constrain each weight vector
        #   of length `(input_dim,)`.
        #   In a `Convolution2D` layer with `data_format="channels_last"`,
        #   the weight tensor has shape
        #   `(rows, cols, input_depth, output_depth)`,
        #   set `axis` to `[0, 1, 2]`
        #   to constrain the weights of each filter tensor of size
        #   `(rows, cols, input_depth)`.

        return w_unit

    def get_config(self):
        config = super(PWMNorm, self).get_config()
        return config

# pwm_norm = PWMNorm
