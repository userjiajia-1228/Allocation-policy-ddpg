
#import tensorflow as tf
import tensorflow.compat.v1 as tf  #tensorflow 如何在2.x 版本 与 1.x 版本间切换的方法
tf.disable_v2_behavior()
import numpy as np
from env import Env
import time
import matplotlib.pyplot as plt
import os

 
 
#####################  hyper parameters  ####################
 
MAX_EPISODES = 30
MAX_EP_STEPS = 3000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic，原来是0.002
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
 
RENDER = False
#ENV_NAME = 'Pendulum-v0'
 
###############################  DDPG  ####################################
 
class TD3(object):
    def __init__(self, s_dim, r_dim, b_dim, o_dim, r_bound, b_bound):
        # dimension
        self.s_dim = s_dim
        self.a_dim = r_dim + b_dim + o_dim
        self.r_dim = r_dim
        self.b_dim = b_dim
        self.o_dim = o_dim
        # self.a_bound
        self.r_bound = r_bound
        self.b_bound = b_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + self.a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.update_cnt = 0     #更新次数
        self.policy_target_update_interval = 3 #策略网络更新频率
        self.sess = tf.Session()
 
        #self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
 
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
            sample = tf.distributions.Normal(loc=0., scale=1.)
            noise = tf.clip_by_value(sample.sample(1) * 0.5, -1, 1)
            noise_a_ = a_ + noise
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q1 = self._build_c(self.S, self.a, scope='eval1', trainable=True)
            q1_ = self._build_c(self.S_, noise_a_, scope='target1', trainable=False)
            q2 = self._build_c(self.S, self.a, scope='eval2', trainable=True)
            q2_ = self._build_c(self.S_, noise_a_, scope='target2', trainable=False)
 
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval1')
        self.ct_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target1')
        self.ce_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval2')
        self.ct_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target2')
        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params1 + self.ct_params2, self.ae_params + self.ce_params1 + self.ce_params2 )]
 
        self.hard_replace = [tf.assign(t, e)
                             for t, e in zip(self.at_params + self.ct_params1 + self.ct_params2, self.ae_params + self.ce_params1 + self.ce_params2 )]
 
        q_target = self.R + GAMMA * tf.minimum(q1_, q2_)
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error1 = tf.losses.mean_squared_error(labels=q_target, predictions=q1)
        self.ctrain1 = tf.train.AdamOptimizer(LR_C).minimize(td_error1, var_list=self.ce_params1)
        td_error2 = tf.losses.mean_squared_error(labels=q_target, predictions=q2)
        self.ctrain2 = tf.train.AdamOptimizer(LR_C).minimize(td_error2, var_list=self.ce_params2)
 
        a_loss = - tf.reduce_mean(q1)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
 
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.hard_replace)  # 初始化目标网络
 
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
 
    def learn(self):
        # soft target replacement
        self.update_cnt += 1
 
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        self.sess.run(self.ctrain1, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        self.sess.run(self.ctrain2, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        if self.update_cnt % self.policy_target_update_interval == 0:
            self.sess.run(self.atrain, {self.S: bs})
            self.sess.run(self.soft_replace)
 
 
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
 
    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            n_l = 50
            net = tf.layers.dense(s, n_l, activation=tf.nn.relu, name='l1', trainable=trainable)
            # resource ( 0 - r_bound)
            layer_r0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='r_0', trainable=trainable)
            layer_r1 = tf.layers.dense(layer_r0, n_l, activation=tf.nn.relu, name='r_1', trainable=trainable)
            layer_r2 = tf.layers.dense(layer_r1, n_l, activation=tf.nn.relu, name='r_2', trainable=trainable)
            layer_r3 = tf.layers.dense(layer_r2, n_l, activation=tf.nn.relu, name='r_3', trainable=trainable)
            layer_r4 = tf.layers.dense(layer_r3, self.r_dim, activation=tf.nn.relu, name='r_4', trainable=trainable)

            # bandwidth ( 0 - b_bound)
            layer_b0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='b_0', trainable=trainable)
            layer_b1 = tf.layers.dense(layer_b0, n_l, activation=tf.nn.relu, name='b_1', trainable=trainable)
            layer_b2 = tf.layers.dense(layer_b1, n_l, activation=tf.nn.relu, name='b_2', trainable=trainable)
            layer_b3 = tf.layers.dense(layer_b2, n_l, activation=tf.nn.relu, name='b_3', trainable=trainable)
            layer_b4 = tf.layers.dense(layer_b3, self.b_dim, activation=tf.nn.relu, name='b_4', trainable=trainable)

            # offloading (probability: 0 - 1)
            # layer
            layer = [["layer" + str(user_id) + str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # name
            name = [["layer" + str(user_id) + str(layer) for layer in range(4)] for user_id in range(self.r_dim)]
            # user
            user = ["user" + str(user_id) for user_id in range(self.r_dim)]
            # softmax
            softmax = ["softmax" + str(user_id) for user_id in range(self.r_dim)]
            for user_id in range(self.r_dim):
                layer[user_id][0] = tf.layers.dense(net, n_l, activation=tf.nn.relu, name=name[user_id][0],
                                                    trainable=trainable)
                layer[user_id][1] = tf.layers.dense(layer[user_id][0], n_l, activation=tf.nn.relu,
                                                    name=name[user_id][1], trainable=trainable)
                layer[user_id][2] = tf.layers.dense(layer[user_id][1], n_l, activation=tf.nn.relu,
                                                    name=name[user_id][2], trainable=trainable)
                layer[user_id][3] = tf.layers.dense(layer[user_id][2], (self.o_dim / self.r_dim), activation=tf.nn.relu,
                                                    name=name[user_id][3], trainable=trainable)
                user[user_id] = tf.nn.softmax(layer[user_id][3], name=softmax[user_id])

            #  concate
            a = tf.concat([layer_r4, layer_b4], 1)
            for user_id in range(self.r_dim):
                a = tf.concat([a, user[user_id]], 1)
            return a
 
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 50
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net, n_l1, activation=tf.nn.relu, trainable=trainable)
            net_2 = tf.layers.dense(net_1, n_l1, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net_2, 1, activation=tf.nn.relu, trainable=trainable)  # Q(s,a)
 
###############################  training  ####################################
TEXT_RENDER = False  # render：提供
SCREEN_RENDER = True
CHANGE = False  # 原来是False
SLEEP_TIME = 0.01  # 原来是0.1
CHECK_EPISODE = 4

#####################  function  ####################
def exploration (a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound#np.random.normal()的意思是一个正态分布，normal这里是正态的意思。
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a  # 数组a中包含了资源和带宽两种数值

###############################  training  ####################################


env = Env()
s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location = env.get_inf()#初始化
#  print(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location)
td3 = TD3(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound)
 
#  var = 3  # control exploration
r_var = 3  # control exploration  ，原来是0.6
b_var = 3
t1 = time.time()
ep_reward = []   # 记录奖励值
r_v, b_v = [], []
var_reward = []
max_rewards = 0
episode = 0
var_counter = 0
epoch_inf = []
while var_counter < MAX_EPISODES:
    s = env.reset()
    ep_reward.append(0)
    
    if SCREEN_RENDER:
        env.initial_screen_demo()
        
    for j in range(MAX_EP_STEPS):
        time.sleep(SLEEP_TIME)  # Python time sleep() 函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起的时间。
        # render
        if SCREEN_RENDER:   # render:提供  ，全局变量SCREEN_RENDER = True
            env.screen_demo()
        if TEXT_RENDER and j % 30 == 0:  # TEXT_RENDER = False#render：提供
            env.text_render()
 
        # Add exploration noise
        a = td3.choose_action(s)
        #  a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        # add randomness to action selection for exploration：将随机性添加到动作选择以进行探索
        a = exploration(a, r_dim, b_dim, r_var, b_var)
        s_, r = env.ddpg_step_forward(a, r_dim, b_dim)  # 带宽，资源，卸载策略等进行相应的更新
 
        td3.store_transition(s, a, r / 10, s_)
 
        if td3.pointer > MEMORY_CAPACITY:
            #  var *= .9995    # decay the action randomness
            td3.learn()
            if CHANGE:
                r_var *= .99999
                b_var *= .99999
 
        s = s_
        ep_reward[episode] += r
        if j == MAX_EP_STEPS-1:
            var_reward.append(ep_reward[episode])
            r_v.append(r_var)
            b_v.append(b_var)
            print('Episode:%3d' % episode, ' Reward: %5d' % ep_reward[episode], '###  r_var: %.2f ' % r_var,'b_var: %.2f ' % b_var, )
            string = 'Episode:%3d' % episode + ' Reward: %5d' % ep_reward[episode] + '###  r_var: %.2f ' % r_var + 'b_var: %.2f ' % b_var
            epoch_inf.append(string)
            # variation change？？？
            if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:#var_counter含义不是太理解
                CHANGE = True  # 全局变量CHECK_EPISODE = 4
                # var_counter = 0
                max_rewards = np.mean(var_reward[-CHECK_EPISODE:])
                var_reward = []
            else:
                CHANGE = False
            var_counter += 1
    # end the episode
    print("用户的延迟数组", env.delay)
    for user in env.U:
        if env.delay[user.user_id] > 1.60:
            env.task_num += 1
    env.text_rate(env.task_num, episode)
    print("任务未完成数量", env.task_num)
    env.task_num = 0
    sum = 0
    for user in env.U:
        sum += env.cost[user.user_id]
    env.text_delay(sum, episode)
    if SCREEN_RENDER:
        env.canvas.tk.destroy()
    episode += 1        
                
print('Running time: ', time.time() - t1)
dir_name = 'D:\download contents\论文\思路论文\Resources-Allocation-in-The-Edge-Computing-Environment-Using-Reinforcement-Learning-master/output-a/' + 'ddpg_'+str(r_dim) + 'u' + str(int(o_dim / r_dim)) + 'e' + str(limit) + 'l' + location
if (os.path.isdir(dir_name)):  # 判断是否是目录
    os.rmdir(dir_name)  # 是，则创建一个目录
os.makedirs(dir_name)
# plot the reward
fig_reward = plt.figure()
plt.plot([i+1 for i in range(len(ep_reward))], ep_reward)
plt.xlabel("episode")
plt.ylabel("rewards")
fig_reward.savefig(dir_name + '/rewards.png')  # 保存奖励图片
# plot the variance
fig_variance = plt.figure()
plt.plot([i + 1 for i in range(len(r_v))], r_v, b_v)  # 原来长度是episode
plt.xlabel("episode")
plt.ylabel("variance")
print('len(U)????', r_dim)
fig_variance.savefig(dir_name + '/variance.png')  # 保存方差图片
# write the record
f = open(dir_name + '/record.txt', 'a')
f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
f.write('代码运行的时间:' + str(time.time() - t1) + '\n\n')
f.write('user_number:' + str(r_dim) + '\n\n')
f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
f.write('limit:' + str(limit) + '\n\n')
f.write('task information:' + '\n')
f.write(task_inf + '\n\n')
for i in range(episode):
    f.write(epoch_inf[i] + '\n')
# mean
print("the mean of the rewards in the last", MAX_EPISODES, " epochs:", str(np.mean(ep_reward[-MAX_EPISODES:])))
f.write("the mean of the rewards:" + str(np.mean(ep_reward[-MAX_EPISODES:])) + '\n\n')
# standard deviation
print("the standard deviation of the rewards:", str(np.std(ep_reward[-MAX_EPISODES:])))
f.write("the standard deviation of the rewards:" + str(np.std(ep_reward[-MAX_EPISODES:])) + '\n\n')
# range
print("the range of the rewards:", str(max(ep_reward[-MAX_EPISODES:]) - min(ep_reward[-MAX_EPISODES:])))
f.write("the range of the rewards:" + str(max(ep_reward[-MAX_EPISODES:]) - min(ep_reward[-MAX_EPISODES:])) + '\n\n')
f.close()


