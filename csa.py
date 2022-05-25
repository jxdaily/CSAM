# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.stats as st
from scipy.misc import imread, imsave

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random

slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'max epsilon.')

tf.flags.DEFINE_integer('num_iter', 10, 'max iteration.')

tf.flags.DEFINE_float('momentum', 1.0, 'momentum about the model.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_float('prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer('image_resize', 331, 'Height of each input images.')

tf.flags.DEFINE_string('checkpoint_path', './models',
                       'Path to checkpoint for pretained models.')

tf.flags.DEFINE_string('input_dir', './dev_data/val_rs',
                       'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './outputs',
                       'Output directory with images.')

FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt')}

def get_expand_imgs(x_test):
    padding = np.zeros(shape=(299,299), dtype='float32')
    x_test_R = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_G = np.zeros(shape=(len(x_test), 299, 299, 3))
    x_test_B = np.zeros(shape=(len(x_test), 299, 299, 3))
    for i in range(len(x_test)):
        R = x_test[i][::,::,0]
        G = x_test[i][::,::,1]
        B = x_test[i][::,::,2]
        x_test_R[i] = np.stack([R, padding, padding], axis=-1)
        x_test_G[i] = np.stack([padding, G, padding], axis=-1)
        x_test_B[i] = np.stack([padding, padding, B], axis=-1)
    return x_test, x_test_R, x_test_G, x_test_B


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            _, tar1, tar2, tar3 = get_expand_imgs(images)
            yield filenames, images, tar1, tar2, tar3
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        _, tar1, tar2, tar3 = get_expand_imgs(images)
        yield filenames, images, tar1, tar2, tar3


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def graph(x, y, i, x_max, x_min, grad, x_tar1, x_tar2, x_tar3):

    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    x_nes = x
    x_res = tf.concat([x_nes, x_nes / 2, x_nes / 4, x_nes / 8, x_nes / 16], axis=0)
    tar1 = tf.concat([x_tar1, x_tar1 / 2, x_tar1 / 4, x_tar1 / 8, x_tar1 / 16], axis=0)
    tar2 = tf.concat([x_tar2, x_tar2 / 2, x_tar2 / 4, x_tar2 / 8, x_tar2 / 16], axis=0)
    tar3 = tf.concat([x_tar3, x_tar3 / 2, x_tar3 / 4, x_tar3 / 8, x_tar3 / 16], axis=0)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            x_res, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3_tar1, end_points_v3_tar1 = inception_v3.inception_v3(
            tar1, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3_tar2, end_points_v3_tar2 = inception_v3.inception_v3(
            tar2, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3_tar3, end_points_v3_tar3 = inception_v3.inception_v3(
            tar3, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    pred = tf.argmax(end_points_v3['Predictions'], 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred[:y.shape[0]] + (1 - first_round) * y
    one_hot = tf.concat([tf.one_hot(y, num_classes)] * 5, axis=0)

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)

    cross_entropy_tar1 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_tar1)

    cross_entropy_tar2 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_tar2)

    cross_entropy_tar3 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_tar3)

    noise = tf.reduce_sum(
        tf.split(tf.gradients(cross_entropy, x_res)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)

    noise += tf.reduce_sum(
        tf.split(tf.gradients(cross_entropy_tar1, tar1)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:,
                                                                 None,
                                                                 None, None, None], axis=0)
    noise += tf.reduce_sum(
        tf.split(tf.gradients(cross_entropy_tar2, tar2)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:,
                                                                 None,
                                                                 None, None, None], axis=0)
    noise += tf.reduce_sum(
        tf.split(tf.gradients(cross_entropy_tar3, tar3)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:,
                                                                 None,
                                                                 None, None, None], axis=0)


    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)

    x_tar1 = x_tar1 + alpha * tf.sign(noise)
    x_tar1 = tf.clip_by_value(x_tar1, x_min, x_max)

    x_tar2 = x_tar2 + alpha * tf.sign(noise)
    x_tar2 = tf.clip_by_value(x_tar2, x_min, x_max)

    x_tar3 = x_tar3 + alpha * tf.sign(noise)
    x_tar3 = tf.clip_by_value(x_tar3, x_min, x_max)

    i = tf.add(i, 1)

    return x, y, i, x_max, x_min, noise, x_tar1, x_tar2, x_tar3


def stop(x, y, i, x_max, x_min, grad, x_tar1, x_tar2, x_tar3):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def main(_):
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_tar1 = tf.placeholder(tf.float32, shape=batch_shape)
        x_tar2 = tf.placeholder(tf.float32, shape=batch_shape)
        x_tar3 = tf.placeholder(tf.float32, shape=batch_shape)

        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad, x_tar1, x_tar2, x_tar3])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            idx = 0
            l2_diff = 0
            for filenames, images, tar1, tar2, tar3 in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))

                adv_images = sess.run(x_adv, feed_dict={x_input: images,
                                                        x_tar1: tar1,
                                                        x_tar2: tar2,
                                                        x_tar3: tar3})

                save_images(adv_images, filenames, FLAGS.output_dir)
                diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))


if __name__ == '__main__':
    tf.app.run()