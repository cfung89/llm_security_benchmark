import tensorflow as tf
print('Tensorflow version:', tf.__version__())
if tf.test.gpu_device_name():
  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
  print("Could not find GPU. Please install GPU version of TF.")
