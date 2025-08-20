# coding=utf-8
"""Implementation of MI-FGSM attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.misc import imread, imsave
from tensorflow.contrib.image import transform as images_transform
from tensorflow.contrib.image import rotate as images_rotate

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

import random

slim = tf.contrib.slim

tf.flags.DEFINE_integer('batch_size', 1, 'How many images process at one time.')

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

tf.flags.DEFINE_string('output_dir', './csamv3',
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


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)

def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    # image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    # out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out

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
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


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

def randx(x, alpha):
    return x + alpha*tf.random_normal(x.shape, mean=0, stddev=1), x - alpha*tf.random_normal(x.shape, mean=0, stddev=1)


def graph(x, x_d1, x_d2, x_d3, x_d4, x_d5, y, i, x_max, x_min, grad):

    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    x_1, x_2 = randx(x, 0.1)
    x_3, x_4 = randx(x, 0.2)
    x_5, x_6 = randx(x, 0.3)
    x_7, x_8 = randx(x, 0.4)
    x_9, x_10 = randx(x, 0.5)
    x_11, x_12 = randx(x, 0.1)

    x_batch = tf.concat([x, x / 2, x / 4, x / 8, x / 16], axis=0)
    x_tar1 = tf.concat([x_d1, x_d1 / 2, x_d1 / 4, x_d1 / 8, x_d1 / 16], axis=0)
    x_tar2 = tf.concat([x_d2, x_d2 / 2, x_d2 / 4, x_d2 / 8, x_d2 / 16], axis=0)
    x_tar3 = tf.concat([x_d3, x_d3 / 2, x_d3 / 4, x_d3 / 8, x_d3 / 16], axis=0)
    x_tar4 = tf.concat([x_d4, x_d4 / 2, x_d4 / 4, x_d4 / 8, x_d4 / 16], axis=0)
    x_tar5 = tf.concat([x_d5, x_d5 / 2, x_d5 / 4, x_d5 / 8, x_d5 / 16], axis=0)
    x1 = tf.concat([x_1, x_1 / 2, x_1 / 4, x_1 / 8, x_1 / 16], axis=0)
    x2 = tf.concat([x_2, x_2 / 2, x_2 / 4, x_2 / 8, x_2 / 16], axis=0)
    x3 = tf.concat([x_3, x_3 / 2, x_3 / 4, x_3 / 8, x_3 / 16], axis=0)
    x4 = tf.concat([x_4, x_4 / 2, x_4 / 4, x_4 / 8, x_4 / 16], axis=0)
    x5 = tf.concat([x_5, x_5 / 2, x_5 / 4, x_5 / 8, x_5 / 16], axis=0)
    x6 = tf.concat([x_6, x_6 / 2, x_6 / 4, x_6 / 8, x_6 / 16], axis=0)
    x7 = tf.concat([x_7, x_7 / 2, x_7 / 4, x_7 / 8, x_7 / 16], axis=0)
    x8 = tf.concat([x_8, x_8 / 2, x_8 / 4, x_8 / 8, x_8 / 16], axis=0)
    x9 = tf.concat([x_9, x_9 / 2, x_9 / 4, x_9 / 8, x_9 / 16], axis=0)
    x10 = tf.concat([x_10, x_10 / 2, x_10 / 4, x_10 / 8, x_10 / 16], axis=0)
    x11 = tf.concat([x_11, x_11 / 2, x_11 / 4, x_11 / 8, x_11 / 16], axis=0)
    x12 = tf.concat([x_12, x_12 / 2, x_12 / 4, x_12 / 8, x_12 / 16], axis=0)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
           x_batch, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_tar1, end_points_v3_tar1 = inception_v3.inception_v3(
            x_tar1, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_tar2, end_points_v3_tar2 = inception_v3.inception_v3(
            x_tar2, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_tar3, end_points_v3_tar3 = inception_v3.inception_v3(
            x_tar3, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_tar4, end_points_v3_tar4 = inception_v3.inception_v3(
            x_tar4, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_tar5, end_points_v3_tar5 = inception_v3.inception_v3(
            x_tar5, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_1, end_points_v3_1 = inception_v3.inception_v3(
            x1, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_2, end_points_v3_2 = inception_v3.inception_v3(
            x2, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_3, end_points_v3_3 = inception_v3.inception_v3(
            x3, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_4, end_points_v3_4 = inception_v3.inception_v3(
            x4, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_5, end_points_v3_5 = inception_v3.inception_v3(
            x5, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_6, end_points_v3_6 = inception_v3.inception_v3(
            x6, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_7, end_points_v3_7 = inception_v3.inception_v3(
            x7, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_8, end_points_v3_8 = inception_v3.inception_v3(
            x8, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_9, end_points_v3_9 = inception_v3.inception_v3(
            x9, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_10, end_points_v3_10 = inception_v3.inception_v3(
            x10, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_11, end_points_v3_11 = inception_v3.inception_v3(
            x11, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)
        logits_v3_12, end_points_v3_12 = inception_v3.inception_v3(
            x12, num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    pred = tf.argmax(end_points_v3['Predictions'], 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred[:y.shape[0]] + (1 - first_round) * y
    one_hot = tf.concat([tf.one_hot(y, num_classes)] * 5, axis=0)

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_v3)
    cross_entropy_tar1 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_tar1)
    cross_entropy_tar2 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_tar2)
    cross_entropy_tar3 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_tar3)
    cross_entropy_tar4 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_tar4)
    cross_entropy_tar5 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_tar5)

    cross_entropy_1 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_1)
    cross_entropy_2 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_2)
    cross_entropy_3 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_3)
    cross_entropy_4 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_4)
    cross_entropy_5 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_5)
    cross_entropy_6 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_6)
    cross_entropy_7 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_7)
    cross_entropy_8 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_8)
    cross_entropy_9 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_9)
    cross_entropy_10 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_10)
    cross_entropy_11 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_11)
    cross_entropy_12 = tf.losses.softmax_cross_entropy(one_hot, logits_v3_12)


    noise = 0.1*tf.reduce_sum(tf.split(tf.gradients(cross_entropy, x_batch)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += 0.1*tf.reduce_sum(tf.split(tf.gradients(cross_entropy_tar1, x_tar1)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += 0.1*tf.reduce_sum(tf.split(tf.gradients(cross_entropy_tar2, x_tar2)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += 0.1*tf.reduce_sum(tf.split(tf.gradients(cross_entropy_tar3, x_tar3)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += 0.1*tf.reduce_sum(tf.split(tf.gradients(cross_entropy_tar4, x_tar4)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += 0.1*tf.reduce_sum(tf.split(tf.gradients(cross_entropy_tar5, x_tar5)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)

    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_1, x1)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_2, x2)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_3, x3)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_4, x4)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_5, x5)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_6, x6)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_7, x7)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_8, x8)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_9, x9)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_10, x10)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_11, x11)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)
    noise += tf.reduce_sum(tf.split(tf.gradients(cross_entropy_12, x12)[0], 5) * tf.constant([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None,
                                                             None, None, None], axis=0)

    # noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)

    x_d1 = x_d1 + alpha * tf.sign(noise)
    x_d1 = tf.clip_by_value(x_d1, x_min, x_max)

    x_d2 = x_d2 + alpha * tf.sign(noise)
    x_d2 = tf.clip_by_value(x_d2, x_min, x_max)

    x_d3 = x_d3 + alpha * tf.sign(noise)
    x_d3 = tf.clip_by_value(x_d3, x_min, x_max)

    x_d4 = x_d4 + alpha * tf.sign(noise)
    x_d4 = tf.clip_by_value(x_d4, x_min, x_max)

    x_d5 = x_d5 + alpha * tf.sign(noise)
    x_d5 = tf.clip_by_value(x_d5, x_min, x_max)

    i = tf.add(i, 1)

    return x, x_d1, x_d2, x_d3, x_d4, x_d5, y, i, x_max, x_min, noise


def stop(x, x_d1, x_d2, x_d3, x_d4, x_d5, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def image_augmentation(x):
    # [32, 299, 299, 3]
    # img, noise
    # batch_size行1列
    # 32,1
    one = tf.fill([tf.shape(x)[0], 1], 1.)
    zero = tf.fill([tf.shape(x)[0], 1], 0.)
    # 32,8
    # 1 , 0, 0, 0 ,1 ,0 ,0, 0
    transforms = tf.concat([one, zero, zero, zero, one, zero, zero, zero], axis=1)
    # 32,6 -> 32,8
    rands = tf.concat([tf.truncated_normal([tf.shape(x)[0], 6], stddev=0.05), zero, zero], axis=1)
    # 32, 32, 8
    return images_transform(x, transforms + rands, interpolation='BILINEAR')


def image_rotation(x):
    """ imgs, scale, scale is in radians """
    rands = tf.truncated_normal([tf.shape(x)[0]], stddev=0.05)
    return images_rotate(x, rands, interpolation='BILINEAR')

def shuffled(images):
    x_d1, x_d2, x_d3, x_d4, x_d5 = np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32'),np.zeros(shape=images.shape, dtype='float32')
    for i in range(len(images)):
        img = images[i]
        R = img[::, ::, 0]
        G = img[::, ::, 1]
        B = img[::, ::, 2]
        x_d1[i] = np.stack([R, B, G], axis=-1)
        x_d2[i] = np.stack([B, G, R], axis=-1)
        x_d3[i] = np.stack([B, R, G], axis=-1)
        x_d4[i] = np.stack([G, R, B], axis=-1)
        x_d5[i] = np.stack([G, B, R], axis=-1)

    return x_d1, x_d2, x_d3, x_d4, x_d5


def input_diversity(input_tensor):
    rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize_images(ret, [FLAGS.image_height, FLAGS.image_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    f2l = load_labels('./dev_data/val_rs.csv')
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_d1 = tf.placeholder(tf.float32, shape=batch_shape)
        x_d2 = tf.placeholder(tf.float32, shape=batch_shape)
        x_d3 = tf.placeholder(tf.float32, shape=batch_shape)
        x_d4 = tf.placeholder(tf.float32, shape=batch_shape)
        x_d5 = tf.placeholder(tf.float32, shape=batch_shape)

        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, x_d1, x_d2, x_d3, x_d4, x_d5, y, i, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])

            idx = 0
            l2_diff = 0
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                idx = idx + 1
                print("start the i={} attack".format(idx))
                img1, img2, img3, img4, img5 = shuffled(images)
                adv_images = sess.run(x_adv, feed_dict={x_input: images,
                                                        x_d1: img1,
                                                        x_d2: img2,
                                                        x_d3: img3,
                                                        x_d4: img4,
                                                        x_d5: img5})

                save_images(adv_images, filenames, FLAGS.output_dir)
                diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            print('{:.2f}'.format(l2_diff * FLAGS.batch_size / 1000))


def load_labels(file_name):
    import pandas as pd
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


if __name__ == '__main__':
    tf.app.run()
