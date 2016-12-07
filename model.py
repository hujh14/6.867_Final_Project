from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from data import Data

class Model():
    def __init__(self, data):
        self.data = data

    def run(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        print "Training..."
        for i in range(20000):
            print i
            x = self.data.images
            masks = self.data.random_masks(x.shape[0])
            if i%100 == 0:
                pred = self.prediction.eval(feed_dict={self.x: x, self.masks: masks})
                for j in xrange(1):
                    self.data.open_image(x[j])
                    self.data.open_image(masks[j])
                    self.data.open_image(pred[j])
            self.train_step.run(feed_dict={self.x: x, self.masks: masks})

        print("test accuracy %g"%self.accuracy.eval(feed_dict={
            self.x: data.test_images, self.masks: data.random_masks()}))

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28])
        self.masks = tf.placeholder(tf.float32, [None, 28, 28])

        masked_x = tf.mul(self.x, self.masks)
        self.y_ = self.x

        x_image = tf.reshape(masked_x, [-1,28,28,1])
        print x_image.get_shape()

        # Encoder 
        layer1_filter_size = 3
        layer1_out_channels = 32
        layer1_out = self.construct_conv_layer(x_image, layer1_filter_size, layer1_out_channels)
        print layer1_out.get_shape()

        layer2_filter_size = 3
        layer2_out_channels = 64
        layer2_out = self.construct_conv_layer(layer1_out, layer2_filter_size, layer2_out_channels)
        print layer2_out.get_shape()

        layer3_filter_size = 3
        layer3_out_channels = 32
        layer3_out = self.construct_conv_layer(layer2_out, layer3_filter_size, layer3_out_channels)
        print layer3_out.get_shape()

        # layer4_filter_size = 3
        # layer4_out_channels = 64
        # layer4_out = self.construct_conv_layer(layer3_out, layer4_filter_size, layer4_out_channels)
        # print layer4_out.get_shape()

        # Channel wise fully connected layer
        channel_out = self.build_channel_wise_fc_layer(layer3_out)
        print channel_out.get_shape()

        # # Decoder
        # layer4_in = self.construct_deconv_layer(channel_out, layer4_filter_size, layer3_out.get_shape().as_list())
        # print layer4_in.get_shape()

        layer3_in = self.construct_deconv_layer(channel_out, layer3_filter_size, layer2_out.get_shape().as_list())
        print layer3_in.get_shape()

        layer2_in = self.construct_deconv_layer(layer3_in, layer2_filter_size, layer1_out.get_shape().as_list())
        print layer2_in.get_shape()

        layer1_in = self.construct_deconv_layer(layer2_in, layer1_filter_size, x_image.get_shape().as_list())
        print layer1_in.get_shape()

        self.y_pred = tf.reshape(layer1_in, [-1,28,28])

        # Define loss and optimizer
        l2_loss = tf.nn.l2_loss(self.y_pred - self.y_)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(l2_loss)

        # correct_prediction = tf.equal(tf.argmax(self.y_pred,1), tf.argmax(self.y_,1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.prediction = self.y_pred

    def build_channel_wise_fc_layer(self, inp):
        _, width, height, n_feat_map = inp.get_shape().as_list()
        inp_reshape = tf.reshape(inp, [-1, width*height, n_feat_map])

        inp_transpose = tf.transpose( inp_reshape, [2,0,1] )

        W = self.weight_variable([n_feat_map,width*height,width*height])
        output = tf.batch_matmul(inp_transpose, W)

        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape

    def construct_conv_layer(self, inp, filter_size, output_channels):
        input_channels = inp.get_shape()[3].value

        W_conv = self.weight_variable([filter_size, filter_size, input_channels, output_channels])
        b_conv = self.bias_variable([output_channels])
        h_conv = tf.nn.relu(self.conv2d(inp, W_conv) + b_conv)
        # print h_conv.get_shape(), "hi"
        # h_pool = self.max_pool_2x2(h_conv)
        # print h_pool.get_shape(), "hi2"
        # return h_pool
        return h_conv

    def construct_deconv_layer(self, out, filter_size, input_shape):
        output_channels = out.get_shape()[3].value
        input_channels = input_shape[3]

        W = self.weight_variable([filter_size, filter_size, input_channels, output_channels])
        b = self.bias_variable([input_channels])
        h = tf.nn.relu(self.deconv2d(out, W, input_shape) + b)
        return h

    def flatten(self, conv):
        shape = conv.get_shape()
        nodes = shape[1].value*shape[2].value*shape[3].value
        return tf.reshape(conv, [-1, nodes])


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def deconv2d(self, x, W, input_shape):
        # Work around
        shape = tf.pack([tf.shape(x)[0], input_shape[1], input_shape[2], input_shape[3]])
        # input_shape[0] = -1
        return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    data = mnist.train.next_batch(1)[0]
    images = data.reshape((1,28,28))
    data = Data(images)

    model = Model(data)
    model.build_model()
    model.run()