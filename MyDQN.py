# -*- coding: UTF-8 -*-
import gym
import tensorflow as tf
import numpy as np
import os
import psutil


class ValueFun():
    """
    Value fiunction, including building model, predict and update
    """

    def __init__(self, env, discount_factor=1, summaries_dir=None):
        self.discount_factor = discount_factor
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.sess = tf.InteractiveSession()
        self.build_model()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        self.sess.run(tf.global_variables_initializer())

        # If saving summaries
        self.summary_writer = None
        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_value_fun")
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

    def build_model(self):

        # set placeholder: input state, input action and output value
        self.state_input = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='state_input')
        self.action_input = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name='action_input')
        self.actual_value = tf.placeholder(shape=[None], dtype=tf.float32, name='actual_value')

        # build network
        self.fc1 = tf.contrib.layers.fully_connected(self.state_input, 20,
                                                     activation_fn=tf.nn.leaky_relu,
                                                     weights_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                                     biases_initializer=tf.zeros_initializer)

        self.value_output = tf.contrib.layers.fully_connected(self.fc1,
                                                              self.action_dim,
                                                              activation_fn=None,
                                                              weights_initializer=tf.random_normal_initializer(mean=0, stddev=1),
                                                              biases_initializer=tf.zeros_initializer)
        # Calculate the loss
        Q_action = tf.reduce_sum(tf.multiply(self.value_output, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.actual_value - Q_action))
        self.optimizer = tf.train.RMSPropOptimizer(0.003, 0.99, 0.0, 1e-6).minimize(self.cost, global_step=tf.train.get_global_step())

        self.summaries = tf.summary.merge([
                    tf.summary.scalar("loss", self.cost),
                    tf.summary.histogram("Q_value", self.value_output),
                    tf.summary.scalar("max_q_value", tf.reduce_max(self.value_output))
                ])

    def predict(self, state, if_debug=False):

        state = np.array(state)

        # Ensure state.ndim == 2
        if state.ndim == 1:
            state = np.expand_dims(state, 0)

        # return the value of each action
        if if_debug:
            return self.sess.run([self.value_output, self.fc1], feed_dict={self.state_input: state})
        else:
            return self.sess.run(self.value_output, feed_dict={self.state_input: state})

    def update(self, state, action, reward, next_state, done):

        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1

        # Calculate target value: V = R + lambda * V_next or V = R if done
        if done:
            target_value = reward
        else:
            next_value = np.max(self.predict(next_state))
            target_value = reward + (self.discount_factor * next_value)

        # Expand dimension 0
        state = np.expand_dims(state, 0)
        action_onehot = np.expand_dims(action_onehot, 0)
        target_value = np.expand_dims(target_value, 0)
        feed_dict = {self.state_input: state, self.action_input: action_onehot, self.actual_value: target_value}

        summaries, _, global_step = self.sess.run(
            [self.summaries, self.optimizer, tf.train.get_global_step()], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

# ValueFun() test:
env = gym.make('CartPole-v0')
value_fun = ValueFun(env, discount_factor=0)
test_state = [1, 1, 1, 1]
value, fc1 = value_fun.predict(test_state, if_debug=True)
print("Value before train : {}".format(value))
print("fc1 : {}".format(fc1))

# Train value function so that the value of action 0 for test_state in 1
for i in range(100):
    value_fun.update(test_state, action=0, reward=20, done=1, next_state=test_state)
    value_fun.update(test_state, action=1, reward=-1, done=1, next_state=test_state)

value, fc1 = value_fun.predict(test_state, if_debug=True)
print("Value after train : {}".format(value))
print("fc1 : {}".format(fc1))


class Policy():

    def __init__(self, value_fun):
        self.value_fun = value_fun

    def action(self, state, epsilon=0.0):
        action_value = self.value_fun.predict(state)
        nA = np.size(action_value)
        best_action = np.argmax(action_value)
        action_prob = np.ones(nA) * epsilon / nA
        action_prob[best_action] += 1 - epsilon
        action = np.random.choice(nA, p=action_prob)
        return action


class Work():

    def __init__(self, env, summaries_dir=None):
        self.env = env
        self.value_fun = ValueFun(env, summaries_dir=summaries_dir)
        self.policy = Policy(self.value_fun)
        self.current_process = psutil.Process()
        self.time = 0

    def train(self, num_episode, num_step):
        for i_episode in range(num_episode):
            state = self.env.reset()
            episode_reward = 0.0
            episode_step = 0.0

            # update parameter each step
            for i_step in range(num_step):
                action = self.policy.action(state, epsilon=0.1)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_step = i_step
                self.value_fun.update(state, action, reward, next_state, done)

                if done:
                    break

                state = next_state

            episode_summary = tf.Summary()
#            episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
            episode_summary.value.add(simple_value=episode_reward, tag="episode/reward")
            episode_summary.value.add(simple_value=episode_step, tag="episode/length")
            episode_summary.value.add(simple_value=self.current_process.cpu_percent(), tag="system/cpu_usage_percent")
            episode_summary.value.add(simple_value=self.current_process.memory_percent(memtype="vms"),
                                      tag="system/v_memeory_usage_percent")

            self.value_fun.summary_writer.add_summary(episode_summary, self.time*num_episode + i_episode)
            self.value_fun.summary_writer.flush()

        self.time += 1

    def test(self, num_episode, num_step, show=True):

        total_reward = 0.0

        for i_episode in range(num_episode):
            state = self.env.reset()

            for i_step in range(num_step):

                if show:
                    self.env.render()

                action = self.policy.action(state, epsilon=0)
                next_state, reward, done, _ = self.env.step(action)
                self.value_fun.update(state, action, reward, next_state, done)
                total_reward += reward

                if done:
                    break

                state = next_state

        return total_reward / num_episode


def main():
    train_episode_each = 100
    train_episode = 10000
    test_episode = 10
    num_step = 300
    env = gym.make('CartPole-v0')
    experiment_dir = os.path.abspath("./MyDQN/{}".format(env.spec.id))
    work = Work(env, summaries_dir=experiment_dir)

    for i in range(train_episode // train_episode_each):
        work.train(num_episode=train_episode_each, num_step=num_step)
        ave_reward = work.test(num_episode=test_episode, num_step=num_step, show=True)
        print("episode: {}, Evaluation Average Reward:{}".format(i*train_episode_each, ave_reward))

    work.test(num_episode=test_episode, num_step=num_step, show=True)


if __name__ == '__main__':
    main()
