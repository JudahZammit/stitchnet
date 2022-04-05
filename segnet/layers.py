import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

class MaxPoolingWithArgmax2D(tfkl.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        argmax = tf.cast(argmax, 'float32')
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

class MaxUnpooling2D(tfkl.Layer):
  def __init__(self, mask_shape, size=(2, 2),**kwargs):
    super(MaxUnpooling2D, self).__init__(**kwargs)
    self.size = size
    self.mask_shape = mask_shape

  def compute_output_shape(self, input_shape):
    return  (self.mask_shape[0],
		     self.mask_shape[1] * self.size[0],
		     self.mask_shape[2] * self.size[1],
		     self.mask_shape[3])

  def call(self, inputs):
    updates, mask = inputs[0], inputs[1]
    mask = tf.cast(mask, "int32")
    input_shape = tf.shape(updates, out_type="int32")
    output_shape = self.compute_output_shape(input_shape)

    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask, dtype="int32")
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
    batch_range = tf.reshape(
        tf.range(output_shape[0], dtype="int32"), shape=batch_shape
    )
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = (mask // output_shape[3]) % output_shape[2]
    feature_range = tf.range(output_shape[3], dtype="int32")
    f = one_like_mask * feature_range

    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(updates)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(updates, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret



class MaxUnpooling2D(tfkl.Layer):
  def __init__(self, size=(2, 2), **kwargs):
    super(MaxUnpooling2D, self).__init__(**kwargs)
    self.size = size

  def call(self, inputs, output_shape=None):
    updates, mask = inputs[0], inputs[1]
    with tf.compat.v1.variable_scope(self.name):
      mask = tf.cast(mask, 'int32')
      input_shape = tf.shape(updates, out_type='int32')
      if output_shape is None:
        output_shape = (
		input_shape[0],
		input_shape[1] * self.size[0],
		input_shape[2] * self.size[1],
		input_shape[3])

      ret = tf.scatter_nd(tfk.backend.expand_dims(tfk.backend.flatten(mask)),
                              tfk.backend.flatten(updates),
                              [tfk.backend.prod(output_shape)])
      input_shape = updates.shape
      out_shape = [-1,
		     input_shape[1] * self.size[0],
		     input_shape[2] * self.size[1],
		     input_shape[3]]
    return tf.reshape(ret, out_shape)


  def compute_output_shape(self, input_shape):
    mask_shape = input_shape[1]
    return (
	    mask_shape[0],
	    mask_shape[1]*self.size[0],
	    mask_shape[2]*self.size[1],
	    mask_shape[3]
	    )

