import tensorflow as tf

model_path = 
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [] #specify optimisation. Generally default
converter.target_spec.supported_types = [] #need to be specified
lite_model = converter.convert()

with open('lite_model', 'w') as f:
  f.write(lite_model)
