# -*- coding: UTF-8 -*-
import gym
import tensorflow as tf
import numpy as np

class ValueFun():

    def __init__(self, env, discount_factor = 1):
        self.discount_factor = discount_factor
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.build_model()

    def predict(self, state):

        # return the value of each action
        return self.sess.run(self.value_output, feed_dict={self.state_input: state})

    def update(self, state, action, reward, next_state, done):

        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1

        if done:
            target_value = reward
        else:
            next_value = np.max(self.predict(next_state))
            target_value = reward + (self.discount_factor * next_value)

        self.sess.run(self.optimizer, feed_dict={self.state_input: state,
                                                 self.action_input: action_onehot,
                                                 self.actual_value: target_value})

    def build_model(self):

        # set placeholder: input state, input action and output value
        self.state_input = tf.placeholder(shape=[self.state_dim], dtype=tf.float32, name='state_input')
        self.action_input = tf.placeholder(shape=[self.action_dim], dtype=tf.float32, name='action_input')
        self.actual_value = tf.placeholder(shape=[self.action_dim], dtype=tf.float32, name='actual_value')

        # build network
        fc1 = tf.contrib.layers.fully_connected(self.state_input, 20, activation_fn=tf.nn.relu)
        self.value_output = tf.contrib.layers.fully_connected(fc1, self.action_dim, activation_fn=tf.nn.relu)

        # Calculate the loss
        Q_action = tf.reduce_sum(tf.multiply(self.value_output, self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.actual_value - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

class Policy():

    def __init__(self, value_fun):
        self.value_fun = value_fun

    def action(self, state, epsilon = 0.0):
        action_value = self.value_fun.predict(state)
        nA = np.size(action_value)
        best_action = np.argmax(action_value)
        action_prob = np.ones(nA) * epsilon / nA
        action_prob[best_action] += 1 - epsilon
        action = np.random.choice(nA, p=action_prob)
        return action

class Work():

    def __init__(self, env):
        self.env = env
        self.value_fun = ValueFun(env)
        self.policy = Policy(self.value_fun)

    def train(self, num_episode, num_step):
        for i_episode in range(num_episode):
            state = self.env.reset()

            # update parameter each step
            for i_step in range(num_step):
                action = self.policy.action(state, epsilon = 0.1)
                next_state, reward, done, _ = self.env.step(action)
                self.value_fun.update(state, action, reward, next_state, done)

                if done:
                    break

                state = next_state

    def test(self, num_episode, num_step, show=True):
        for i_episode in range(num_episode):
            state = self.env.reset()

            for i_step in range(num_step):

                if show:
                    self.env.render()

                action = self.policy.action(state, epsilon=0)
                next_state, reward, done, _ = self.env.step(action)

                if done:
                    break

                state = next_state

def main():
    train_episode_each = 100
    train_episode = 10000
    test_episode = 10
    num_step = 300
    env = gym.make('CartPole-v0')
    work = Work(env)

    for i in range(train_episode // train_episode_each):
        work.train(num_episode=train_episode_each, num_step=num_step)
        work.test(num_episode=test_episode, num_step=num_step, show=True)

    work.test(num_episode=test_episode, num_step=num_step, show=True)

if __name__ == '__main__':
    main()
