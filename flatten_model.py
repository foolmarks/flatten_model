import tensorflow as tf


def flatten_nested_model(model):
  """Flatten the nested model."""
  def flatten_model(model, inp):
    out = inp
    for layer in model.layers:
      if isinstance(layer, tf.keras.layers.InputLayer):
        continue
      elif isinstance(layer, tf.keras.Model):
        print("Flatten nested model: ", layer.name)
        out = flatten_model(layer, out)
      else:
        out = layer(out)
    return out
  out = flatten_model(model, model.input)
  return tf.keras.Model(model.input, out)

model = tf.keras.models.load_model('float/f_model.h5')

flat_model = flatten_nested_model(model)
flat_model.summary()

flat_model.save('float/flat_float_model.h5')
