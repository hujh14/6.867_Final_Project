from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class Model():
    def __init__(self):
        pass
    def run(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        for i in range(20000):
          batch = mnist.train.next_batch(50)
          if i%100 == 0:
            train_accuracy = self.accuracy.eval(feed_dict={
                self.x:batch[0], self.y_: batch[1]})
            print("step %d, training accuracy %g"%(i, train_accuracy))
          self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})

        print("test accuracy %g"%self.accuracy.eval(feed_dict={
            self.x: mnist.test.images, self.y_: mnist.test.labels}))

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        x_image = tf.reshape(self.x, [-1,28,28,1])

        feature_rep = self.build_downsampling_model(x_image)
        upsampling_inp = self.build_channel_wise_fc_layer(feature_rep)

        feature_rep_flat = self.flatten(upsampling_inp)

        fc_layer_hidden_units = 1024
        fc_layer_out = self.construct_fc_layer(feature_rep_flat, fc_layer_hidden_units)

        output_units = 10
        self.y_pred = self.construct_fc_layer(fc_layer_out, output_units, RELU=False)

        self.y_ = tf.placeholder(tf.float32, [None, 10])

        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_pred, self.y_))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_pred,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def build_downsampling_model(self, inp):
        layer1_filter_size = 5
        layer1_channels = 32
        layer1_out = self.construct_conv_layer(inp, layer1_filter_size, layer1_channels)

        layer2_filter_size = 7
        layer2_channels = 64
        layer2_out = self.construct_conv_layer(layer1_out, layer2_filter_size, layer2_channels)

        return layer2_out

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
        h_pool = self.max_pool_2x2(h_conv)
        return h_pool

    def construct_fc_layer(self, inp, hidden_units, RELU=True):
        input_nodes = inp.get_shape()[1].value
        W_fc = self.weight_variable([input_nodes, hidden_units])
        b_fc = self.bias_variable([hidden_units])
        h_fc = tf.matmul(inp, W_fc) + b_fc
        if RELU:
            return tf.nn.relu(h_fc)
        else:
            return h_fc

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

    def max_pool_2x2(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

model = Model()
model.build_model()
model.run()