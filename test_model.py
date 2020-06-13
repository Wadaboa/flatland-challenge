from collections import deque
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import random
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras.layers.core import Activation
from keras.utils import np_utils, to_categorical
from keras.engine.topology import Layer

neighbors = 4
len_feature = 35
action_space = 4


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done, adjacency):
        experience = (state, action, reward, new_state, done, adjacency)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


def Adjacency(railway_encoding, n_agents):
    adj = []
    for j in range(n_agents):
        neigh = railway_encoding.get_neighboring_agents(j)
        l = to_categorical(neigh, num_classes=n_agents)
        for i in range(4):
            if neigh[i] == -1:
                l[i] = np.zeros(n_agents)
        adj.append(l)
    return adj


def MLP():

    In_0 = Input(shape=[len_feature])
    h = Dense(128, activation='relu', kernel_initializer='random_normal')(In_0)
    h = Dense(128, activation='relu', kernel_initializer='random_normal')(h)
    h = Reshape((1, 128))(h)
    model = Model(input=In_0, output=h)
    return model


def MultiHeadsAttModel(l=2, d=128, dv=16, dout=128, nv=8):

    v1 = Input(shape=(l, d))
    q1 = Input(shape=(l, d))
    k1 = Input(shape=(l, d))
    ve = Input(shape=(1, l))

    v2 = Dense(dv*nv, activation="relu",
               kernel_initializer='random_normal')(v1)
    q2 = Dense(dv*nv, activation="relu",
               kernel_initializer='random_normal')(q1)
    k2 = Dense(dv*nv, activation="relu",
               kernel_initializer='random_normal')(k1)

    v = Reshape((l, nv, dv))(v2)
    q = Reshape((l, nv, dv))(q2)
    k = Reshape((l, nv, dv))(k2)
    v = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(v)
    k = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(k)
    q = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(q)

    att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[
                 3, 3]) / np.sqrt(dv))([q, k])  # l, nv, nv
    att_ = Lambda(lambda x: K.softmax(x))(att)
    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3, 2]))([att, v])
    out = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(out)

    out = Reshape((l, dv*nv))(out)

    T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([ve, out])

    out = Dense(dout, activation="relu", kernel_initializer='random_normal')(T)
    model = Model(inputs=[q1, k1, v1, ve], outputs=out)
    model_ = Model(inputs=[q1, k1, v1, ve], outputs=att_)
    return model, model_


def Q_Net(action_dim):

    I1 = Input(shape=(1, 128))
    I2 = Input(shape=(1, 128))
    I3 = Input(shape=(1, 128))

    h1 = Flatten()(I1)
    h2 = Flatten()(I2)
    h3 = Flatten()(I3)

    h = Concatenate()([h1, h2, h3])
    V = Dense(action_dim, kernel_initializer='random_normal')(h)

    model = Model(input=[I1, I2, I3], output=V)
    return model


######build the model#########
encoder = MLP()
m1, m1_r = MultiHeadsAttModel(l=neighbors)
m2, m2_r = MultiHeadsAttModel(l=neighbors)
q_net = Q_Net(action_dim=action_space)
vec = np.zeros((1, neighbors))
n_agents = 3
vec[0][0] = 1

In = []
for j in range(n_agents):
    In.append(Input(shape=[len_feature]))
    In.append(Input(shape=(neighbors, n_agents)))
In.append(Input(shape=(1, neighbors)))
feature = []
for j in range(n_agents):
    feature.append(encoder(In[j*2]))

feature_ = Concatenate(axis=1)(feature)

relation1 = []
for j in range(n_agents):
    T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([In[j*2+1], feature_])
    relation1.append(m1([T, T, T, In[n_agents*2]]))

relation1_ = Concatenate(axis=1)(relation1)

relation2 = []
for j in range(n_agents):
    T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([In[j*2+1], relation1_])
    relation2.append(m2([T, T, T, In[n_agents*2]]))

V = []
for j in range(n_agents):
    V.append(q_net([feature[j], relation1[j], relation2[j]]))

model = Model(input=In, output=V)
model.compile(optimizer=Adam(lr=0.0001), loss='mse')

######build the target model#########
encoder_t = MLP()
m1_t = MultiHeadsAttModel(l=neighbors)
m2_t = MultiHeadsAttModel(l=neighbors)
q_net_t = Q_Net(action_dim=action_space)
In_t = []
for j in range(n_agents):
    In_t.append(Input(shape=[len_feature]))
    In_t.append(Input(shape=(neighbors, n_agents)))
In_t.append(Input(shape=(1, neighbors)))

feature_t = []
for j in range(n_agents):
    feature_t.append(encoder_t(In_t[j*2]))

feature_t_ = Concatenate(axis=1)(feature_t)

relation1_t = []
for j in range(n_agents):
    T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([In_t[j*2+1], feature_t_])
    relation1_t.append(m1_t([T, T, T, In_t[n_agents*2]]))

relation1_t_ = Concatenate(axis=1)(relation1_t)

relation2_t = []
for j in range(n_agents):
    T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([In_t[j*2+1], relation1_t_])
    relation2_t.append(m2_t([T, T, T, In_t[n_agents*2]]))

V_t = []
for j in range(n_agents):
    V_t.append(q_net_t([feature_t[j], relation1_t[j], relation2_t[j]]))

model_t = Model(input=In_t, output=V_t)


capacity = 200000
TAU = 0.01
alpha = 0.6
GAMMA = 0.98
episode_before_train = 20
num_episodes = 100
i_episode = 0
mini_batch = 10
loss, score = 0, 0
num = 0
times = [0]*n_agents
total_time = 0
buff = ReplayBuffer(capacity)


def train(env):
    while i_episode < num_episodes:
        i_episode += 1
        obs, info = env.reset(True, True)
        adj = Adjacency(env.obs_builder.railway_encoding, n_agents)
        max_steps = int(3 * (env.height + env.width))

        # Run episode
        for step in range(max_steps):

            ob = []
            for j in range(n_agents):
                ob.append(np.asarray([obs[j]]))
                ob.append(np.asarray([adj[j]]))
            ob.append(np.asarray([vec]))
            action = model.predict(ob)
            act = np.zeros(n_agents, dtype=np.int32)
            for j in range(n_agents):
                if np.random.rand() < alpha:
                    act[j] = random.randrange(action_space)
                else:
                    act[j] = np.argmax(action[j])

            next_obs, reward, done, _ = env.step(act)
            buff.add(obs, act, next_obs, reward, done, adj)
            score += sum(reward)
            obs = next_obs.copy()
            adj = Adjacency(env.obs_builder.railway_encoding, n_agents)

            if done['__all__']:
                break

            if i_episode % 100 == 0:
                print(int(i_episode/100))
                print(score/100, end='\t')
                if num != 0:
                    print(total_time/num, end='\t')
                else:
                    print(0, end='\t')
                print(num, end='\t')
                print(loss/100)
                loss = 0
                score = 0
                num = 0
                total_time = 0

            if i_episode < episode_before_train:
                continue

            #########training#########
            batch = buff.getBatch(mini_batch)
            states, actions, rewards, new_states, dones = [], [], [], [], []
            for i_ in range(n_agents*2+1):
                states.append([])
                new_states.append([])
                for e in batch:
                    for j in range(n_agents):
                        states[j*2].append(e[0][j])
                        states[j*2+1].append(e[5][j])
                        new_states[j*2].append(e[2][j])
                        new_states[j*2+1].append(e[5][j])
                    states[n_agents*2].append(vec)
                    new_states[n_agents*2].append(vec)
                    actions.append(e[1])
                    rewards.append(e[3])
                    dones.append(e[4])

                actions = np.asarray(actions)
                rewards = np.asarray(rewards)
                dones = np.asarray(dones)

                for i_ in range(n_agents*2+1):
                    states[i_] = np.asarray(states[i_])
                    new_states[i_] = np.asarray(new_states[i_])

                q_values = model.predict(states)
                target_q_values = model_t.predict(new_states)
                for k in range(len(batch)):
                    for j in range(n_agents):
                        if dones[k][j]:
                            q_values[j][k][actions[k][j]] = rewards[k][j]
                        else:
                            q_values[j][k][actions[k][j]] = rewards[k][j] + \
                                GAMMA*np.max(target_q_values[j][k])

                history = model.fit(
                    states, q_values, epochs=1, batch_size=10, verbose=0)
                his = 0
                for (k, v) in history.history.items():
                    his += v[0]
                loss += (his/n_agents)

                #########training target model#########
                weights = encoder.get_weights()
                target_weights = encoder_t.get_weights()
                for w in range(len(weights)):
                    target_weights[w] = TAU * weights[w] + \
                        (1 - TAU) * target_weights[w]
                encoder_t.set_weights(target_weights)

                weights = q_net.get_weights()
                target_weights = q_net_t.get_weights()
                for w in range(len(weights)):
                    target_weights[w] = TAU * weights[w] + \
                        (1 - TAU) * target_weights[w]
                q_net_t.set_weights(target_weights)

                weights = m1.get_weights()
                target_weights = m1_t.get_weights()
                for w in range(len(weights)):
                    target_weights[w] = TAU * weights[w] + \
                        (1 - TAU) * target_weights[w]
                m1_t.set_weights(target_weights)

                weights = m2.get_weights()
                target_weights = m2_t.get_weights()
                for w in range(len(weights)):
                    target_weights[w] = TAU * weights[w] + \
                        (1 - TAU) * target_weights[w]
                m2_t.set_weights(target_weights)

                model.save('dgn.h5')
