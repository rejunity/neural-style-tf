import tensorflow as tf

def searchsortedN(sorted_array, values_to_search, n):
  from tensorflow.python.ops import array_ops
  
  indices = array_ops.zeros_like(values_to_search, dtype=tf.float32)
  n = int(n)
  
  while n > 1:
    n = n / 2

    idxL = indices
    idxR = indices + tf.to_float(n)

    pred = tf.less(values_to_search, tf.gather(sorted_array, tf.to_int32(idxR)))
    indices = tf.where(pred, idxL, idxR)

  pred = tf.less(values_to_search, sorted_array[0])
  indices = tf.where(pred, indices, indices + 1)
  return indices

def interp_linear(x_new, x, y, nbins):
  from tensorflow.python.framework import dtypes
  from tensorflow.python.ops import clip_ops
  from tensorflow.python.ops import math_ops

  x_new_indices = searchsortedN(x, x_new, nbins)

  lo = x_new_indices - 1
  hi = x_new_indices

  # Clip indices so that they are within the range
  hi = math_ops.cast(
    clip_ops.clip_by_value(hi, 0, nbins-1), dtypes.int32)
  lo = math_ops.cast(
    clip_ops.clip_by_value(lo, 0, nbins-1), dtypes.int32)

  x_lo = tf.gather(x, lo) #x_lo = x[lo]
  x_hi = tf.gather(x, hi) #x_hi = x[hi]
  y_lo = tf.gather(y, lo) #y_lo = y[lo]
  y_hi = tf.gather(y, hi) #y_hi = y[hi]

  # Calculate the slope of regions that each x_new value falls in.
  dx = (x_hi - x_lo)
  slope = (y_hi - y_lo) / dx

  # Calculate the actual value for each entry in x_new.
  y_linear = slope*(x_new - x_lo) + y_lo
  y_nearest = y_lo

  # Protect against NaN (div-by-zero)
  p = tf.not_equal(dx, 0.0)
  y_new = tf.where(p, y_linear, y_nearest)

  return y_new

def histogram_fixed_width(values, value_range, nbins=100):
  from tensorflow.python.framework import dtypes
  from tensorflow.python.ops import clip_ops
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops import array_ops

  nbins_float = math_ops.to_float(nbins)

  # Map tensor values that fall within value_range to [0, 1].
  scaled_values = math_ops.truediv(values - value_range[0],
                  value_range[1] - value_range[0],
                  name='scaled_values')

  # map tensor values within the open interval value_range to {0,.., nbins-1},
  # values outside the open interval will be zero or less, or nbins or more.
  indices = math_ops.floor(nbins_float * scaled_values, name='indices')

  # Clip edge cases (e.g. value = value_range[1]) or "outliers."
  indices = math_ops.cast(
    clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)

  #counts = tf.Variable(...) <= array_ops.zeros_like(indices, dtype=dtypes.int32))
  #return tf.scatter_add(counts, indices, array_ops.ones_like(indices, dtype=dtypes.int32)), indices

  return math_ops.unsorted_segment_sum(
    array_ops.ones_like(indices, dtype=dtypes.float32),
    indices,
    nbins), indices

FEATURE_GAMMA = 4.0
def compress(x, value_range):
  x = (x-value_range[0]) / (value_range[1] - value_range[0]) # remove bias and normalize
  x = tf.pow(x, 1/FEATURE_GAMMA)
  return x

def decompress(x, value_range):
  x = tf.pow(x, FEATURE_GAMMA)
  x = x * (value_range[1] - value_range[0]) + value_range[0]
  return x

def match_histogram(source, template, nbins):

  value_range = [tf.reduce_min(template), tf.reduce_max(template)]
  source = compress(source, value_range)
  template = compress(template, value_range)

  t_value_range = [tf.reduce_min(template), tf.reduce_max(template)]
  s_value_range = [tf.reduce_min(source), tf.reduce_max(source)]
  #s_value_range = value_range
  #t_value_range = value_range
  t_values = tf.linspace(t_value_range[0], t_value_range[1], nbins)

  s_counts, indices = histogram_fixed_width(source, s_value_range, nbins)
  t_counts, _ = histogram_fixed_width(template, t_value_range, nbins)

  s_cdf = tf.to_float(s_counts)
  s_cdf = tf.cumsum(s_cdf)
  s_cdf /= s_cdf[-1]

  t_cdf = tf.to_float(t_counts)
  t_cdf = tf.cumsum(t_cdf)
  t_cdf /= t_cdf[-1]

  interp_t_values = interp_linear(s_cdf, t_cdf, t_values, nbins)
  interp_t_values = tf.maximum(interp_t_values, 0.0)
  values = tf.gather(interp_t_values, indices)

  values = decompress(values, value_range)

  return values

def match_histogram_featurewise(a, x, nbins):
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import control_flow_ops
  from tensorflow.python.ops import tensor_array_ops

  n = array_ops.shape(a)[0]
  ta = tensor_array_ops.TensorArray(dtype=a.dtype, size=n, dynamic_size=False, infer_shape=True)

  def compute(i, ta):
    x_matched_to_a = match_histogram(x[i], a[i], nbins)
    ta = ta.write(i, x_matched_to_a)
    return (i + 1, ta)

  i = tf.constant(0)
  _, res = control_flow_ops.while_loop(
    lambda i, _: i < n, compute, (i, ta))

  return res.stack()
