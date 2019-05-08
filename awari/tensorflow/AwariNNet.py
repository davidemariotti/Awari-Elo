import sys
import traceback
sys.path.append('..')
from utils import *

import tensorflow as tf

class nn_helper():
    def __init__(self):
        self.hist_added = 0

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.variable_scope('summaries'):
            mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.variable_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding scope ensures logical grouping of the layers in the graph.
        with tf.variable_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.variable_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights)
            with tf.variable_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases)
            with tf.variable_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def add_hist(self, train_vars):
        for i in train_vars:
            name = i.name.split(":")[0]
            value = i.value()
            if self.hist_added < 150:
                print('add histogram: %s %s' % (name, value))
                tf.summary.histogram(name, value)
            else:
                print('do not add histogram: %s %s' % (name, value))

class AwariNNet():
    def __init__(self, game, args):
        # game params
        # self.board_x, self.board_y = game.getBoardSize()
        # Awari mod:
        self.board_x, self.board_y, self.image_stack_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # tensorboard
        self.nn_helper = nn_helper()

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default(): 
            # self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])    # s: batch_size x board_x x board_y
            # Awari mod:
            # tensorboard:
            with tf.variable_scope('input'):
                # self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y, self.image_stack_size], name = 'board-input')    # s: batch_size x board_x x board_y
                # self.dropout = tf.placeholder(tf.float32, name = 'input-drop')
                self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y, self.image_stack_size])    # s: batch_size x board_x x board_y
                self.dropout = tf.placeholder(tf.float32)
                tf.summary.scalar('dropout_keep_probability', self.dropout)
                # self.isTraining = tf.placeholder(tf.bool, name="is_training")
                self.isTraining = tf.placeholder(tf.bool)

            # x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])                    # batch_size  x board_x x board_y x 1
            # Awari mod:
            with tf.variable_scope('input_reshape'):
                x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, self.image_stack_size])                    # batch_size  x board_x x board_y x 1
                # tf.summary.image('input', x_image, 10)
            with tf.variable_scope('conv2d'):
                h_conv1 = Relu(BatchNormalization(self.conv2d(x_image, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
                h_conv2 = Relu(BatchNormalization(self.conv2d(h_conv1, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
                # h_conv3 = Relu(BatchNormalization(self.conv2d(h_conv2, args.num_channels, 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-2) x (board_y-2) x num_channels
                # h_conv4 = Relu(BatchNormalization(self.conv2d(h_conv3, args.num_channels, 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-4) x (board_y-4) x num_channels
                # Awari mods:
                h_conv3 = Relu(BatchNormalization(self.conv2d(h_conv2, args.num_channels, 'same'), axis=3, training=self.isTraining))    # batch_size  x (board_x-2) x (board_y-2) x num_channels
                h_conv4 = Relu(BatchNormalization(self.conv2d(h_conv3, args.num_channels, 'same'), axis=3, training=self.isTraining))    # batch_size  x (board_x-4) x (board_y-4) x num_channels
            with tf.variable_scope('output_reshape'):
                # h_conv4_flat = tf.reshape(h_conv4, [-1, args.num_channels*(self.board_x-4)*(self.board_y-4)])
                # Awari mod:
                h_conv4_flat = tf.reshape(h_conv4, [-1, args.num_channels*self.board_x*self.board_y])
            with tf.variable_scope('dense_output'):
                s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024, use_bias=False), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
                s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512, use_bias=False), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
                self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
                self.prob = tf.nn.softmax(self.pi)
                self.v = Tanh(Dense(s_fc2, 1))                                                               # batch_size x 1

            self.calculate_loss()

            train_vars = tf.trainable_variables()
            self.nn_helper.add_hist(train_vars)

            # Merge all the summaries
            self.summary_merged = tf.summary.merge_all()

    def conv2d(self, x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding, use_bias=False)

    def calculate_loss(self):
        with tf.variable_scope('output_loss'):
            self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
            self.target_vs = tf.placeholder(tf.float32, shape=[None])
        with tf.variable_scope('cross_entropy'):
            self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        tf.summary.scalar('loss_pi', self.loss_pi)
        with tf.variable_scope('mse'):
            self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
            self.total_loss = self.loss_pi + self.loss_v
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        tf.summary.scalar('loss_v', self.loss_v)
        with tf.variable_scope('train'):
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)

class ResNet():

    def __init__(self, game, args):
        # game params
        # self.board_x, self.board_y = game.getBoardSize()
        # Awari mod:
        self.board_x, self.board_y, self.image_stack_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # tensorboard
        self.nn_helper = nn_helper()

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default(): 
            # self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])    # s: batch_size x board_x x board_y
            # Awari mod:
            with tf.variable_scope('input'):
                self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y, self.image_stack_size], name = "boards")    # s: batch_size x board_x x board_y
                self.dropout = tf.placeholder(tf.float32)
                tf.summary.scalar('dropout_keep_probability', self.dropout)
                self.isTraining = tf.placeholder(tf.bool, name="is_training")

            # x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])                    # batch_size  x board_x x board_y x 1
            # Awari mod:
            with tf.variable_scope('reshape'):
                x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, self.image_stack_size])                    # batch_size  x board_x x board_y x 1
                # x_image = tf.layers.conv2d(x_image, args.num_channels, kernel_size=(3, 3), strides=(1, 1),name='conv',padding='same',use_bias=False)
                x_image = tf.layers.conv2d(x_image, args.num_channels, kernel_size=(3, 3), strides=(1, 1),padding='same',use_bias=False)
                # x_image = tf.layers.batch_normalization(x_image, axis=1, name='conv_bn', training=self.isTraining)
                x_image = tf.layers.batch_normalization(x_image, axis=1, training=self.isTraining)
                x_image = tf.nn.relu(x_image)

            with tf.variable_scope('res-tower'):
                residual_tower = self.residual_block(inputLayer=x_image, kernel_size=3, filters=args.num_channels, stage=1, block='a')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=2, block='b')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=3, block='c')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=4, block='d')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=5, block='e')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=6, block='g')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=7, block='h')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=8, block='i')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=9, block='j')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=10, block='k')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=11, block='m')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=12, block='n')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=13, block='o')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=14, block='p')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=15, block='q')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=16, block='r')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=17, block='s')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=18, block='t')
                residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args.num_channels, stage=19, block='u')

            with tf.variable_scope('policy-out'):
                # policy = tf.layers.conv2d(residual_tower, 2,kernel_size=(1, 1), strides=(1, 1),name='pi',padding='same',use_bias=False)
                policy = tf.layers.conv2d(residual_tower, 2,kernel_size=(1, 1), strides=(1, 1),padding='same',use_bias=False)
                # policy = tf.layers.batch_normalization(policy, axis=3, name='bn_pi', training=self.isTraining)
                policy = tf.layers.batch_normalization(policy, axis=3, training=self.isTraining)
                policy = tf.nn.relu(policy)
                # policy = tf.layers.flatten(policy, name='p_flatten')
                policy = tf.layers.flatten(policy)
                self.pi = tf.layers.dense(policy, self.action_size)
                self.prob = tf.nn.softmax(self.pi)

            with tf.variable_scope('value-out'):
                # value = tf.layers.conv2d(residual_tower, 1,kernel_size=(1, 1), strides=(1, 1),name='v',padding='same',use_bias=False)
                value = tf.layers.conv2d(residual_tower, 1,kernel_size=(1, 1), strides=(1, 1),padding='same',use_bias=False)
                # value = tf.layers.batch_normalization(value, axis=3, name='bn_v', training=self.isTraining)
                value = tf.layers.batch_normalization(value, axis=3, training=self.isTraining)
                value = tf.nn.relu(value)
                # value = tf.layers.flatten(value, name='v_flatten')
                value = tf.layers.flatten(value)
                value = tf.layers.dense(value, units=256)
                value = tf.nn.relu(value)
                value = tf.layers.dense(value, 1)
                self.v = tf.nn.tanh(value) 

            self.calculate_loss()

            train_vars = tf.trainable_variables()
            self.nn_helper.add_hist(train_vars)

            self.summary_merged = tf.summary.merge_all()

    def residual_block(self,inputLayer, filters,kernel_size,stage,block):
        conv_name = 'res' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'

        shortcut = inputLayer

        with tf.variable_scope('res-%d' % stage):
            # residual_layer = tf.layers.conv2d(inputLayer, filters,kernel_size=(kernel_size, kernel_size), strides=(1, 1),name=conv_name+'2a',padding='same',use_bias=False)
            residual_layer = tf.layers.conv2d(inputLayer, filters,kernel_size=(kernel_size, kernel_size), strides=(1, 1),padding='same',use_bias=False)
            # residual_layer = tf.layers.batch_normalization(residual_layer, axis=3, name=bn_name+'2a', training=self.isTraining)
            residual_layer = tf.layers.batch_normalization(residual_layer, axis=3, training=self.isTraining)
            residual_layer = tf.nn.relu(residual_layer)
            # residual_layer = tf.layers.conv2d(residual_layer, filters,kernel_size=(kernel_size, kernel_size), strides=(1, 1),name=conv_name+'2b',padding='same',use_bias=False)
            residual_layer = tf.layers.conv2d(residual_layer, filters,kernel_size=(kernel_size, kernel_size), strides=(1, 1),padding='same',use_bias=False)
            # residual_layer = tf.layers.batch_normalization(residual_layer, axis=3, name=bn_name+'2b', training=self.isTraining)
            residual_layer = tf.layers.batch_normalization(residual_layer, axis=3, training=self.isTraining)
            add_shortcut = tf.add(residual_layer, shortcut)
            residual_result = tf.nn.relu(add_shortcut)
        
        return residual_result

    def calculate_loss(self):
        with tf.variable_scope('output_loss'):
            self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
            self.target_vs = tf.placeholder(tf.float32, shape=[None])
        with tf.variable_scope('cross_entropy'):
            self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        tf.summary.scalar('loss_pi', self.loss_pi)
        with tf.variable_scope('mse'):
            self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
            self.total_loss = self.loss_pi + self.loss_v
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        tf.summary.scalar('loss_v', self.loss_v)
        with tf.variable_scope('train'):
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)


