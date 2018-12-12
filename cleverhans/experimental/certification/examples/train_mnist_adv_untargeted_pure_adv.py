"""
Coverage project model for MNIST
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import tensorflow as tf
from tensorflow.python.platform import flags
import time
import random

from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.dataset import MNIST
from cleverhans.evaluation import accuracy
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils_tf import infer_devices
from cleverhans.serial import save
from cleverhans.serial import NoRefModel

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

epoch = 0
best_result = 0.
best_epoch = -1
last_test_print = time.time()
last_train_print = time.time()

assert __file__.startswith('train_')
assert __file__.endswith('.py')
SAVE_PATH = __file__[len('train_'):-len(".py")] + ".joblib"
TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000
NB_EPOCHS = 350
devices = infer_devices()
num_devices = len(devices)
BATCH_SIZE = 64 * num_devices
LEARNING_RATE = 5e-4
USE_EMA = 1
EMA_DECAY = 'ema_decay_orig'
NB_FILTERS = 32
NCHW, NHWC = 'NCHW NHWC'.split()


def downscale2d(x, n=2, order=NHWC):
  """Box downscaling.

  Args:
    x: 4D tensor in order format.
    n: integer scale.
    order: enum(NCHW, NHWC), the order of channels vs dimensions.

  Returns:
    4D tensor down scaled by a factor n.

  Raises:
    ValueError: if order not NCHW or NHWC.
  """
  if order not in (NCHW, NHWC):
    raise ValueError('Unsupported tensor order %s' % order)
  if n <= 1:
    return x
  if order == NCHW:
    pool2, pooln = [1, 1, 2, 2], [1, 1, n, n]
  else:
    pool2, pooln = [1, 2, 2, 1], [1, n, n, 1]
  if n % 2 == 0:
    x = tf.nn.avg_pool(x, pool2, pool2, 'VALID', order)
    return downscale2d(x, n // 2, order)
  return tf.nn.avg_pool(x, pooln, pooln, 'VALID', order)


class Model(NoRefModel):
  """
  Simple convnet model.
  """

  def __init__(self, scales=2, filters=32, scope=None,
               nb_classes=10, input_shape=(None, 28, 28, 1), **kwargs):
    if scope is None:
      scope = 'deep_classify_' + hex(random.getrandbits(128))[2:-1]
    self.input_shape = input_shape
    self.scales = scales
    self.filters = filters
    super(type(self), self).__init__(nb_classes=nb_classes,
                                     needs_dummy_fprop=True,
                                     scope=scope, **kwargs)

  def fprop(self, x,
            **kwargs):
    scales = self.scales
    filters = self.filters
    scope = self.scope
    act = tf.nn.leaky_relu
    conv_args = dict(kernel_size=3, activation=act,
                     padding='same', data_format='channels_last')
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      y = tf.layers.conv2d(x, filters, **conv_args)
      for scale in range(scales):
        y = tf.layers.conv2d(y, filters << scale, **conv_args)
        y = tf.layers.conv2d(y, filters << (scale + 1), **conv_args)
        y = downscale2d(y, 2)
      assert self.nb_classes == 10
      y = tf.layers.conv2d(y, self.nb_classes, 3)
      logits = tf.reduce_mean(y, [1, 2])
      return {'logits': logits}

  def make_input_placeholder(self):
    return tf.placeholder(tf.float32, self.input_shape, 'x')

  def make_label_placeholder(self):
    return tf.placeholder(tf.float32, (None, self.nb_classes))


def ema_decay_orig(epoch, batch):

  def return_0():
    return 0.

  def return_999():
    return .999

  def return_9999():
    return .9999

  def inner_cond():
    return tf.cond(epoch < 100, return_999, return_9999)

  out = tf.cond(epoch < 10, return_0, inner_cond)
  return out


def ema_decay_2(epoch, batch):
  def return_0():
    return 0.

  def return_9999():
    return .9999

  out = tf.cond(epoch < 30, return_0, return_9999)
  return out


def do_train(train_start=TRAIN_START, train_end=60000, test_start=0,
             test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
             learning_rate=LEARNING_RATE,
             backprop_through_attack=False,
             nb_filters=NB_FILTERS,
             num_threads=None,
             use_ema=USE_EMA,
             ema_decay=EMA_DECAY):
  print('Parameters')
  print('-'*79)
  for x, y in sorted(locals().items()):
    print('%-32s %s' % (x, y))
  print('-'*79)

  if os.path.exists(FLAGS.save_path):
    print(
        "Model " + FLAGS.save_path + " already exists. Refusing to overwrite.")
    quit()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  dataset = MNIST(train_start=train_start, train_end=train_end,
                  test_start=test_start, test_end=test_end,
                  center=True)

  # Use Image Parameters
  img_rows, img_cols, nchannels = dataset.x_train.shape[1:4]
  nb_classes = dataset.NB_CLASSES

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  train_params = {
      'nb_epochs': nb_epochs,
      'learning_rate': learning_rate,
      'batch_size': batch_size,
  }
  eval_params = {'batch_size': batch_size}
  rng = np.random.RandomState([2017, 8, 30])
  sess = tf.Session()

  def do_eval(x_set, y_set, is_adv=None):
    acc = accuracy(sess, model, x_set, y_set)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'clean'
    if report_text:
      print('Accuracy on %s examples: %0.4f' % (report_text, acc))
    return acc

  model = Model(filters=nb_filters)
  model.dataset_factory = dataset.get_factory()

  pgd = ProjectedGradientDescent(model=model, sess=sess)

  center = dataset.kwargs['center']
  value_range = 1. + center
  base_eps = 8. / 255.

  attack_params = {
      'eps': base_eps * value_range,
      'clip_min': -float(center),
      'clip_max': float(center),
      'eps_iter': (2. / 255.) * value_range,
      'nb_iter': 40.
  }

  loss = CrossEntropy(
      model,
      attack=pgd,
      adv_coeff=1.,
      attack_params=attack_params,
  )

  print_test_period = 10
  print_train_period = 50

  def evaluate():
    global epoch
    global last_test_print
    global last_train_print
    global best_result
    global best_epoch
    with sess.as_default():
        print("Saving to ", FLAGS.save_path)
        save(FLAGS.save_path, model)
    if epoch % print_test_period == 0 or time.time() - last_test_print > 300:
      t1 = time.time()
      result = do_eval(dataset.x_test, dataset.y_test, False)
      t2 = time.time()
      if result >= best_result:
        if result > best_result:
          best_epoch = epoch
        else:
          # Keep track of ties
          assert result == best_result
          if not isinstance(best_epoch, list):
            if best_epoch == -1:
              best_epoch = []
            else:
              best_epoch = [best_epoch]
          best_epoch.append(epoch)
        best_result = result
      print("Best so far: ", best_result)
      print("Best epoch: ", best_epoch)
      last_test_print = t2
      print("Test eval time: ", t2 - t1)
    if (epoch % print_train_period == 0 or
        time.time() - last_train_print > 3000):
      t1 = time.time()
      print("Training set: ")
      do_eval(dataset.x_train, dataset.y_train, False)
      t2 = time.time()
      print("Train eval time: ", t2 - t1)
      last_train_print = t2
    epoch += 1

  optimizer = None

  ema_decay = globals()[ema_decay]
  assert callable(ema_decay)

  train(sess, loss, dataset.x_train, dataset.y_train, evaluate=evaluate,
        optimizer=optimizer,
        args=train_params, rng=rng, var_list=model.get_params(),
        use_ema=use_ema, ema_decay=ema_decay)
  # Make sure we always evaluate on the last epoch, so pickling bugs are more
  # obvious
  if (epoch - 1) % print_test_period != 0:
    do_eval(dataset.x_test, dataset.y_test, False)
  if (epoch - 1) % print_train_period != 0:
    print("Training set: ")
    do_eval(dataset.x_train, dataset.y_train, False)

  with sess.as_default():
    save(FLAGS.save_path, model)
    # Now that the model has been saved, you can evaluate it in a
    # separate process using `evaluate_pickled_model.py`.
    # You should get exactly the same result for both clean and
    # adversarial accuracy as you get within this program.


def main(argv=None):
  if len(argv) > 1:
    raise ValueError("Unparsed arguments to script: ", argv[1:])
  do_train(train_end=FLAGS.train_end, test_end=FLAGS.test_end,
           nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
           learning_rate=FLAGS.learning_rate,
           backprop_through_attack=FLAGS.backprop_through_attack,
           use_ema=FLAGS.use_ema,
           ema_decay=FLAGS.ema_decay,
           nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
  flags.DEFINE_string('save_path', SAVE_PATH, 'Path to save to')
  flags.DEFINE_integer('train_end', TRAIN_END,
                       'Ending index of range of training examples to use')
  flags.DEFINE_integer('test_end', TRAIN_END,
                       'Ending index of range of test examples to use')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('use_ema', USE_EMA, 'Whether to use EMA')
  flags.DEFINE_string('ema_decay', EMA_DECAY,
                      'Name of function to use for EMA decay schedule')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Number of filters in convolutions.')
  flags.DEFINE_bool('backprop_through_attack', False,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))

  tf.app.run()
