import tensorflow as tf
import numpy as np

class Model():
    def __init__(self):
        self.layer1_filter_size = 3
        self.layer1_out_channels = 16
        self.layer2_filter_size = 3
        self.layer2_out_channels = 16
        self.layer3_filter_size = 3
        self.layer3_out_channels = 16

        self.build_model()
        self.saver = tf.train.Saver()

    def train(self, iterations, data, mask_gen, model_name):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print "Training..."
            for i in range(iterations):
                x = data.train_images
                masks = mask_gen.random_masks(x.shape[0])
                self.train_step.run(feed_dict={self.x: x, self.masks: masks})

                # Print average pixel
                n = min(x.shape[0], 10)
                print i, self.accuracy.eval(feed_dict={self.x:x[:n], self.masks: mask_gen.center_masks(n, 10)})
            print "Done."

            model_path = self.saver.save(sess, "models/{}.ckpt".format(model_name))
            return model_path

    def test(self, model_path, data, mask_gen):
        ''' mask is one mask applied to all the data
        '''
        with tf.Session() as sess:
            # Restore variables from disk.
            self.saver.restore(sess, model_path)
            print("Model restored.")

            x = data.one_of_each()
            x = data.getNumber(1,7)
            # x = data.test_images
            for i in xrange(1):
                masks = mask_gen.center_masks(x.shape[0], 24)

                pred = self.prediction.eval(feed_dict={self.x: x, self.masks: masks})
                inputs = []
                outputs = []
                for j in xrange(x.shape[0]):
                # for j in xrange(3):
                    inp = np.multiply(x[j], masks[j])
                    inputs.append(inp)
                    outputs.append(pred[j])
                    # q.append(inp)
                    # w.append(pred[j])
                mask_gen.open_images(inputs, outputs)
            # mask_gen.open_images(q,w)


    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28])
        self.masks = tf.placeholder(tf.float32, [None, 28, 28])

        masked_x = tf.mul(self.x, self.masks)
        self.y_ = self.x

        x_image = tf.reshape(masked_x, [-1,28,28,1])
        print x_image.get_shape()

        # Encoder 
        layer1_out = self.construct_conv_layer(x_image, self.layer1_filter_size, self.layer1_out_channels)
        print layer1_out.get_shape()

        layer2_out = self.construct_conv_layer(layer1_out, self.layer2_filter_size, self.layer2_out_channels)
        print layer2_out.get_shape()

        layer3_out = self.construct_conv_layer(layer2_out, self.layer3_filter_size, self.layer3_out_channels)
        print layer3_out.get_shape()

        # layer4_out = self.construct_conv_layer(layer3_out, self.layer4_filter_size, self.layer4_out_channels)
        # print layer4_out.get_shape()

        # Channel wise fully connected layer
        channel_out = self.build_channel_wise_fc_layer(layer3_out)
        print channel_out.get_shape()

        # # Decoder
        # layer4_in = self.construct_deconv_layer(channel_out, self.layer4_filter_size, layer3_out.get_shape().as_list())
        # print layer4_in.get_shape()

        layer3_in = self.construct_deconv_layer(channel_out, self.layer3_filter_size, layer2_out.get_shape().as_list())
        print layer3_in.get_shape()

        layer2_in = self.construct_deconv_layer(layer3_in, self.layer2_filter_size, layer1_out.get_shape().as_list())
        print layer2_in.get_shape()

        layer1_in = self.construct_deconv_layer(layer2_in, self.layer1_filter_size, x_image.get_shape().as_list())
        print layer1_in.get_shape()


        # Feed in or not
        self.y_pred = tf.reshape(layer1_in, [-1,28,28])

        # recon = tf.reshape(layer1_in, [-1,28,28])
        # W1 = self.weight_variable([1,28,28])
        # W2 = self.weight_variable([1,28,28])
        # self.y_pred = tf.mul(W1, recon) + tf.mul(W2, masked_x)

        # output = tf.reshape(layer1_in, [-1,28,28])
        # self.max_val = tf.reduce_max(output)
        # self.y_pred = (1.0/self.max_val)*output

        loss = self.y_pred - self.y_
        masked_loss = tf.mul(1-self.masks, loss)

         # Choose between loss or masked loss
        l2_loss = tf.nn.l2_loss(masked_loss)

        self.train_step = tf.train.AdamOptimizer(4e-3).minimize(l2_loss)
        self.accuracy = tf.reduce_mean(l2_loss)
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

        W = self.weight_variable([filter_size, filter_size, input_channels, output_channels])
        # b = self.bias_variable([output_channels])
        b = tf.zeros([output_channels])
        h = tf.nn.relu(self.conv2d(inp, W) + b)
        # print h.get_shape(), "hi"
        # h_pool = self.max_pool_2x2(h)
        # print h_pool.get_shape(), "hi2"
        # return h_pool
        return h

    def construct_deconv_layer(self, out, filter_size, input_shape):
        output_channels = out.get_shape()[3].value
        input_channels = input_shape[3]

        W = self.weight_variable([filter_size, filter_size, input_channels, output_channels])
        # b = self.bias_variable([input_channels])
        b = tf.zeros([input_channels])
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