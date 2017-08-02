from __future__ import print_function
from helpers import merge, count_params, cache_result
from random import randint
from zap50k import zap_data, IMAGE_SIZE
import itertools
import json
import math
import numpy as np
import os
import scipy.misc
import tensorflow as tf
import time

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

TINY = 1e-8

#########
# Flags #
#########

flags = tf.app.flags
flags.DEFINE_string("file_pattern", "ut-zap50k-images/*/*/*/*.jpg", "Pattern to find zap50k images")
flags.DEFINE_string("logdir", None, "Directory to save logs")
flags.DEFINE_integer("epochs", 1, "Number of epochs")
flags.DEFINE_integer("task", 0, "Task index")
flags.DEFINE_integer("num_gpus", 0, "GPU count")
flags.DEFINE_string("role","worker", "Role")
flags.DEFINE_string("worker_hosts","localhost:2222", "Workers")
flags.DEFINE_string("ps_hosts","localhost:2222", "Ps")
flags.DEFINE_boolean("classifier", False, "Use the discriminator for classification")
flags.DEFINE_boolean("kmeans", False, "Run kmeans of intermediate features")
flags.DEFINE_boolean("similarity", False, "Find most similar shoe")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS


##################
# Model settings #
##################

Z_DIM = 80
C_DIM = 8
C_COEFF = .05

##########
# Models #
##########


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generator(z, latent_c):
    depths = [32, 64, 64, 64, 64, 64, 3]
    sizes = zip(
        np.linspace(4, IMAGE_SIZE['resized'][0], len(depths)).astype(np.int),
        np.linspace(6, IMAGE_SIZE['resized'][1], len(depths)).astype(np.int))
    with slim.arg_scope([slim.conv2d_transpose],
                        normalizer_fn=slim.batch_norm,
                        kernel_size=3):
        with tf.variable_scope("gen"):
            size = sizes.pop(0)
            net = tf.concat(axis=1, values=[z, latent_c])
            net = slim.fully_connected(net, depths[0] * size[0] * size[1])
            net = tf.reshape(net, [-1, size[0], size[1], depths[0]])
            for depth in depths[1:-1] + [None]:
                net = tf.image.resize_images(
                    net, sizes.pop(0),
                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                if depth:
                    net = slim.conv2d_transpose(net, depth)
            net = slim.conv2d_transpose(
                net, depths[-1], activation_fn=tf.nn.tanh, stride=1, normalizer_fn=None)
            tf.summary.image("gen", net, max_outputs=8)
    return net


def discriminator(input, reuse, dropout, int_feats=False, c_dim=None):
    depths = [16 * 2**x for x in range(5)] + [16]
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        reuse=reuse,
                        normalizer_fn=slim.batch_norm,
                        activation_fn=lrelu):
        with tf.variable_scope("discr"):
            net = input
            for i, depth in enumerate(depths):
                if i != 0:
                    #                    net = slim.dropout(net, dropout, scope='dropout')
                    net = slim.dropout(net, 0.5, scope='dropout')
                if i % 2 == 0:
                    net = slim.conv2d(
                        net, depth, kernel_size=3, stride=2, scope='conv%d' % i)
                else:
                    net = slim.conv2d(
                        net, depth, kernel_size=3, scope='conv%d' % i)
            net = slim.flatten(net)
            if int_feats:
                return net
            else:
                d_net = slim.fully_connected(
                    net, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None, scope='out')
    if c_dim:
        with tf.variable_scope('latent_c'):
            q_net = slim.fully_connected(
                net, c_dim, activation_fn=tf.nn.tanh, scope='out')
        return d_net, q_net
    return d_net


def loss(d_model, g_model, dg_model, q_model, latent_c):
    t_vars = tf.trainable_variables()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Latent_C
    q_loss = tf.reduce_sum(0.5 * tf.square(latent_c - q_model)) * C_COEFF

    # Discriminator
    d_loss = -tf.reduce_mean(tf.log(d_model + TINY) + tf.log(1. - dg_model + TINY))
    tf.summary.scalar('d_loss', d_loss)
    d_trainer = tf.train.AdamOptimizer(.0002, beta1=.5).minimize(
        d_loss + q_loss,
        global_step=global_step,
        var_list=[v for v in t_vars if 'discr/' in v.name or 'latent_c/' in v.name])

    # Generator
    g_loss = -tf.reduce_mean(tf.log(dg_model + TINY))
    tf.summary.scalar('g_loss', g_loss)
    g_trainer = tf.train.AdamOptimizer(.001, beta1=.5).minimize(
        g_loss + q_loss,
        var_list=[v for v in t_vars if 'gen/' in v.name or 'latent_c/' in v.name])

    return d_trainer, d_loss, g_trainer, g_loss, global_step


#######
# GAN #
#######


def gan(cluster):
    # Model
    is_chief = (FLAGS.task == 0)
    local_step = 0
    server = tf.train.Server(
        cluster, job_name='worker', task_index=FLAGS.task)
    if FLAGS.num_gpus>0:
        worker_device = '/job:worker/task:%d/gpu:0' % (FLAGS.task)
    else:
        worker_device = '/job:worker/task:%d/cpu:0' % (FLAGS.task)
    with tf.device('/cpu:0'), tf.Session() as cpu_sess:
        dataset = zap_data(FLAGS, True)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.train.start_queue_runners(sess=cpu_sess)
        num_global = (dataset['size'] / FLAGS.batch_size) * FLAGS.epochs

        with tf.device(
            tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device='/job:ps/cpu:0',
                cluster=cluster)),tf.Session():
            x = tf.placeholder(tf.float32, shape=[
                None, IMAGE_SIZE['resized'][0], IMAGE_SIZE['resized'][1], 3])
            dropout = tf.placeholder(tf.float32)
            d_model = discriminator(x, reuse=False, dropout=dropout)

            z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
            latent_c = tf.placeholder(shape=[None, C_DIM], dtype=tf.float32)
            g_model = generator(z, latent_c)
            dg_model, q_model = discriminator(
                g_model, reuse=True, dropout=dropout, c_dim=C_DIM)

            d_trainer, d_loss, g_trainer, g_loss, global_step = loss(
                d_model, g_model, dg_model, q_model, latent_c)

            # Stats
            t_vars = tf.trainable_variables()
            count_params(t_vars, ['discr/', 'gen/', 'latent_c/'])
            # for v in t_vars:
            # tf.histogram_summary(v.name, v)

            # Init
            summary = tf.summary.merge_all()

            init_op = tf.global_variables_initializer()

            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=FLAGS.logdir,
                init_op=init_op,
                recovery_wait_secs=1,
                summary_op=None,
                global_step=global_step)

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task])

            step = 0
            with sv.managed_session(server.target, config=sess_config) as sess:
                # Dataset queue
                while step < num_global and not sv.should_stop():
                    z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, Z_DIM]).astype(np.float32)
                    c_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, C_DIM])
                    images, _ = cpu_sess.run(dataset['batch'])
                    feed_dict = {z: z_batch, latent_c: c_batch, x: images, dropout: .5, }

                    # Update discriminator
                    start = time.time()
                    _, d_loss_val = sess.run([d_trainer, d_loss], feed_dict=feed_dict)
                    d_time = time.time() - start

                    # Update generator
                    start = time.time()
                    _, g_loss_val, summary_str, step = sess.run([g_trainer, g_loss, summary,global_step], feed_dict=feed_dict)
                    g_time = time.time() - start

                    # Log details
                    if local_step % 10 == 0:
                        print("[%s, %s] Disc loss: %.3f (%.2fs), Gen Loss: %.3f (%.2fs)" %
                              (step, step * FLAGS.batch_size / dataset['size'], d_loss_val, d_time, g_loss_val, g_time, ))
                        if is_chief:
                            sv.summary_computed(sess,summary_str)

                    local_step += 1
                    # Early stopping
                    if np.isnan(g_loss_val) or np.isnan(d_loss_val):
                        print('Early stopping', g_loss_val, d_loss_val)
                        break

                    # Finish off the filename queue coordinator.
                coord.request_stop()
                coord.join(threads)
                return


########
# Main #
########

def main(_):
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({
        'ps': ps_spec,
        'worker': worker_spec})
    if FLAGS.role == 'ps':
        print('Start parameter server %d' % (FLAGS.task))
        server = tf.train.Server(
            cluster, job_name=FLAGS.role, task_index=FLAGS.task)
        server.join()
        return
    if not tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.MakeDirs(FLAGS.logdir)
    gan(cluster)


if __name__ == '__main__':
    tf.app.run()
