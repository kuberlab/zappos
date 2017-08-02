from __future__ import print_function
import numpy as np
import tensorflow as tf
from main import discriminator
from zap50k import zap_data, IMAGE_SIZE
import itertools
import scipy.misc


flags = tf.app.flags
flags.DEFINE_string("file_pattern", "ut-zap50k-images/*/*/*/*.jpg", "Pattern to find zap50k images")
flags.DEFINE_string("logdir", None, "Directory to save logs")
flags.DEFINE_string("test_file", None, "File for test")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS

def serving(FLAGS, sess):
    with sess.as_default():
        dataset = zap_data(FLAGS, False)
        x = tf.placeholder(tf.float32, shape=[
            None, IMAGE_SIZE['resized'][0], IMAGE_SIZE['resized'][1], 3])
        dropout = tf.placeholder(tf.float32)
        feat_model = discriminator(x, reuse=False, dropout=dropout, int_feats=True)

        all_features = np.zeros((dataset['size'], feat_model.get_shape()[1]))
        x1 = tf.placeholder(tf.float32, shape=[None, all_features.shape[1]])
        x2 = tf.placeholder(tf.float32, shape=[None, all_features.shape[1]])
        l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=1))

        # Init
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
        saver.restore(sess, checkpoint)

        all_paths = []
        for i in itertools.count():
            try:
                images, paths = sess.run(dataset['batch'])
            except tf.errors.OutOfRangeError:
                break
            if i % 10 == 0:
                print(i * FLAGS.batch_size, dataset['size'])
            im_features = sess.run(feat_model, feed_dict={x: images, dropout: 1, })
            all_features[FLAGS.batch_size * i:FLAGS.batch_size * i + im_features.shape[0]] = im_features
            all_paths += list(paths)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)
        clip = 1e-3
        np.clip(all_features, -clip, clip, all_features)

    def similarity(img,size):
        X = np.zeros([1,IMAGE_SIZE['resized'][0], IMAGE_SIZE['resized'][1], 3], dtype=np.float32)
        X[0] = img
        test_feature = sess.run(feat_model, feed_dict={x: X, dropout: 1, })
        def select_images(distances):
            indices = np.argsort(distances)
            images = []
            for i in range(size):
                images += [dict(path=all_paths[indices[i]],
                            index=indices[i],
                            distance=distances[indices[i]])]
            return images
        bs = 100
        item_block = np.reshape(np.tile(test_feature[0], bs), [bs, -1])
        distances = np.zeros(all_features.shape[0])
        for i in range(0, all_features.shape[0], bs):
            if i + bs > all_features.shape[0]:
                bs = all_features.shape[0] - i
            distances[i:i + bs] = sess.run(l2diff, feed_dict={x1: item_block[:bs], x2: all_features[i:i + bs]})
        return select_images(distances)
    return similarity

def main(_):
    img = scipy.misc.imread(FLAGS.test_file, mode='RGB')
    img = scipy.misc.imresize(img, IMAGE_SIZE['resized'])
    img = img * (1. / 255) - 0.5
    with tf.Session() as sess:
        similarity = serving(FLAGS,sess)
        res = similarity(img,10)
        print(res)


if __name__ == '__main__':
    tf.app.run()


