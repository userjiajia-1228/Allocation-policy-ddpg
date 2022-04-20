import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
from keras.layers import Reshape, Dense, Input, Flatten, Conv2D, MaxPooling2D, Embedding, CuDNNLSTM, LSTM, Activation, RepeatVector, Permute
from keras.layers import Multiply


TIME_STEPS = 20
USER_NUM = 10
#####################  hyper parameters  ####################
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 32
OUTPUT_GRAPH = False

##################################################


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[1])
    # print("input_dim"+str(input_dim))
    a = RepeatVector(12)(inputs)
    a = Permute((2, 1))(a)
    #  a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([a_probs, inputs])
    return output_attention_mul


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, s_dim, r_dim, b_dim, o_dim, r_bound, b_bound):
        self.memory_capacity = 10000
        # dimension
        self.s_dim = s_dim
        self.a_dim = r_dim + b_dim + o_dim
        self.r_dim = r_dim
        self.b_dim = b_dim
        self.o_dim = o_dim
        # self.a_bound
        self.r_bound = r_bound
        self.b_bound = b_bound
        # S, S_, R
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        # memory
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + self.a_dim + 1), dtype=np.float32)  # s_dim + a_dim + r + s_dim
        self.pointer = 0
        # session
        self.sess = tf.Session()

        # define the input and output
        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )

        # replaced target parameters with the trainning  parameters for every step
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))
        # update the weight for every step
        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        # Actor learn()
        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        # Critic learn()
        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def noisy_dense(self, units, input):  #输出的神经元大小
        w_shape = [units, input.shape[1].value]
        mu_w = tf.Variable(initial_value=tf.truncated_normal(shape=w_shape))
        sigma_w = tf.Variable(initial_value=tf.constant(0.017, shape=w_shape))
        epsilon_w = tf.random_uniform(shape=w_shape)

        b_shape = [units]
        mu_b = tf.Variable(initial_value=tf.truncated_normal(shape=b_shape))
        sigma_b = tf.Variable(initial_value=tf.constant(0.017, shape=b_shape))
        epsilon_b = tf.random_uniform(shape=b_shape)

        w = tf.add(mu_w, tf.multiply(sigma_w, epsilon_w))
        b = tf.add(mu_b, tf.multiply(sigma_b, epsilon_b))
        return tf.matmul(input, tf.transpose(w)) + b

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            n_l = 50
            net = tf.layers.dense(s, n_l, activation=tf.nn.relu, name='l1', trainable=trainable)
            # resource ( 0 - r_bound)
            '''layer_r0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='r_0', trainable=trainable)
            layer_r1 = tf.layers.dense(layer_r0, n_l, activation=tf.nn.relu, name='r_1', trainable=trainable)
            # layer_r2 = tf.layers.dense(layer_r1, n_l, activation=tf.nn.relu, name='r_2', trainable=trainable)
            layer_r2 = tf.nn.relu(self.noisy_dense(n_l, layer_r1))
            layer_r3 = tf.layers.dense(layer_r2, n_l, activation=tf.nn.relu, name='r_3', trainable=trainable)
            layer_r4 = tf.layers.dense(layer_r3, self.r_dim, activation=tf.nn.relu, name='r_4', trainable=trainable)'''

            out = Embedding(USER_NUM, USER_NUM)(s)  # 嵌入层USER_NUM = 10
            # s = tf.reshape(s, (3, ))
            # s = tf.expand_dims(s, axis=1)
            out = LSTM(USER_NUM, return_sequences=False)(out)
            out = Activation('softmax')(out)  # softmax
            out = RepeatVector(12)(out)  # 将输入重复n次，原先2D张量，尺寸为(num_samples,features),之后3D张量，尺寸为(num_samples,n,features)
            out = Permute([2, 1])(out)  # permute(dims)：将tensor的维度换位。
            out = LSTM(USER_NUM, return_sequences=False)(out)
            out = attention_3d_block(out)
            out = Dense(24, activation="relu")(s)
            out = Dense(24, activation="relu")(out)  # 24
            layer_r4 = Dense(self.r_dim, activation="relu")(out)

            # bandwidth ( 0 - b_bound)
            '''layer_b0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='b_0', trainable=trainable)
            layer_b1 = tf.layers.dense(layer_b0, n_l, activation=tf.nn.relu, name='b_1', trainable=trainable)
            # layer_b2 = tf.layers.dense(layer_b1, n_l, activation=tf.nn.relu, name='b_2', trainable=trainable)
            layer_b2 = tf.nn.relu(self.noisy_dense(n_l, layer_b1))
            layer_b3 = tf.layers.dense(layer_b2, n_l, activation=tf.nn.relu, name='b_3', trainable=trainable)
            layer_b4 = tf.layers.dense(layer_b3, self.b_dim, activation=tf.nn.relu, name='b_4', trainable=trainable)'''

            out = Embedding(USER_NUM, USER_NUM)(s)  # 嵌入层
            out = LSTM(USER_NUM, return_sequences=False)(out)
            out = Activation('softmax')(out)
            out = RepeatVector(12)(out)  # 将输入重复n次，原先2D张量，尺寸为(num_samples,features),之后3D张量，尺寸为(num_samples,n,features)
            out = Permute([2, 1])(out)  # permute(dims)：将tensor的维度换位。
            out = LSTM(USER_NUM, return_sequences=False)(out)
            out = attention_3d_block(out)
            out = Dense(24, activation="relu")(s)
            out = Dense(24, activation="relu")(out)
            layer_b4 = Dense(self.b_dim, activation="relu")(out)

            # offloading (probability: 0 - 1)
            # layer
            layer = [["layer"+str(user_id)+str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # name
            name = [["layer"+str(user_id) + str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # user
            user = ["user"+str(user_id) for user_id in range(self.r_dim)]
            # softmax
            softmax = ["softmax"+str(user_id) for user_id in range(self.r_dim)]
            for user_id in range(self.r_dim):
                layer[user_id][0] = tf.layers.dense(net, n_l, activation=tf.nn.relu, name=name[user_id][0], trainable=trainable)
                layer[user_id][1] = tf.layers.dense(layer[user_id][0], n_l, activation=tf.nn.relu, name=name[user_id][1], trainable=trainable)
                layer[user_id][2] = tf.layers.dense(layer[user_id][1], n_l, activation=tf.nn.relu, name=name[user_id][2], trainable=trainable)
                # layer[user_id][2] = tf.nn.relu(self.noisy_dense(n_l, layer[user_id][1]))
                layer[user_id][3] = tf.layers.dense(layer[user_id][2], (self.o_dim/self.r_dim), activation=tf.nn.relu, name=name[user_id][3], trainable=trainable)
                user[user_id] = tf.nn.softmax(layer[user_id][3], name=softmax[user_id])

            #   concate
            a = tf.concat([layer_r4, layer_b4], 1)
            # a = tf.reshape(a, tf.shape(user[self.r_dim-1]))
            for user_id in range(self.r_dim):
                # tf.reshape(user[user_id], tf.shape(a))
                a = tf.concat([a, user[user_id]], 1)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        # Q value (0 - inf)
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l = 50
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l], trainable=trainable)
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # net_2 = tf.layers.dense(net_1, n_l, activation=tf.nn.relu, trainable=trainable)
            '''net_2 = tf.nn.relu(self.noisy_dense(n_l, net_1))
            net_3 = tf.layers.dense(net_2, n_l, activation=tf.nn.relu, trainable=trainable)
            net_4 = tf.layers.dense(net_3, n_l, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net_4, 1, activation=tf.nn.relu, trainable=trainable) ''' # Q(s,a)
            out = Embedding(USER_NUM, USER_NUM)(net_1)  # 嵌入层
            out = LSTM(USER_NUM, return_sequences=False)(out)
            out = Activation('softmax')(out)
            out = RepeatVector(12)(out)  # 将输入重复n次，原先2D张量，尺寸为(num_samples,features),之后3D张量，尺寸为(num_samples,n,features)
            out = Permute([2, 1])(out)  # permute(dims)：将tensor的维度换位。
            out = LSTM(USER_NUM, return_sequences=False)(out)
            out = attention_3d_block(out)
            out = Dense(24, activation="relu")(net_1)
            out = Dense(24, activation="relu")(out)
            net_5 = Dense(1, activation="relu")(out)
            return net_5

