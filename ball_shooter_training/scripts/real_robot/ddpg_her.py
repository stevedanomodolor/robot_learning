'''
DDPG reinforcement learning approach

Inspired by https://keras.io/examples/rl/ddpg_pendulum/

        @author: Stevedan Ogochukwu Omodolor "stevedan.ogochu.omodolor@estudiantat.upc.edu"

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.getcwd()
#import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.3, dt=1, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self,num_states , num_actions, buffer_capacity=100000, batch_size=64, gamma = 0.02):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        self.gamma = gamma

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))


    # Takes (s,a,r,s') obervation tuple as input
    def record(self, prev_state, action, reward, state):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        # #print("-----------------------------------------------")
        # #print(str(prev_state))
        # #print(str(action))
        # #print(str(reward))
        # #print(str(state))
        # #print(str(self.state_buffer))
        # #print(str(self.action_buffer))
        # #print(str(self.reward_buffer))
        # #print(str(self.next_state_buffer))
        # #print("-----------------------------------------------")
        # #print(str(len(self.state_buffer)))
        # #print(str(len(self.action_buffer)))
        # #print(str(len(self.reward_buffer)))
        # #print(str(len(self.next_state_buffer)))
        self.state_buffer[index] = prev_state #obs_tuple[0]
        self.action_buffer[index] = action[0] #obs_tuple[1]
        self.reward_buffer[index] = reward #obs_tuple[2]
        self.next_state_buffer[index] = state #obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,actor_model, critic_model, target_actor, target_critic
    , actor_optimizer, critic_optimizer):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self,actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch,actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer)


class DDPGHER:
    def __init__(self, n_actions, n_states, lower_bound , upper_bound , noise_std_dev, critic_lr, actor_lr, buffer_capacity, batch_size, tau, gamma, use_model,model_path, model_actor, model_critic, model_actor_target,model_critic_target ):
        # initiing variables
        self.n_actions = n_actions
        self.n_states = n_states
        self.action_lower_bound = lower_bound
        self.action_upper_bound = upper_bound
        self.std_dev = noise_std_dev
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        std_test = self.std_dev * np.ones(self.n_actions)
        #print("noise deviation: " + str(std_test))
        self.noise_object = OUActionNoise(mean=np.zeros(self.n_actions), std_deviation=self.std_dev * np.ones(self.n_actions))
        self.use_model = use_model
        self.model_path = model_path
        self.model_actor = model_path+model_actor
        self.model_critic = model_path+model_critic
        self.model_actor_target = model_path+model_actor_target
        self.model_critic_target = model_path+model_critic_target
        self.actor_model = self.get_actor(self.model_actor)
        self.critic_model = self.get_critic(self.model_critic)

        self.target_actor = self.get_actor(self.model_actor_target)
        self.target_critic = self.get_critic(self.model_critic_target)

        # Making the weights equal initially
        if not self.use_model:
            self.target_actor.set_weights(self.actor_model.get_weights())
            self.target_critic.set_weights(self.critic_model.get_weights())

        #applying learning rates
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        # initialling buffer
        self.buffer = Buffer(num_states = self.n_states , num_actions = self.n_actions, buffer_capacity=self.buffer_capacity, batch_size=self.batch_size, gamma = self.gamma)

    def save_model_weigth(self, model_path, model_name):
        # model_path = os.getcwd() + "./src/ball_shooter/ball_shooter_training/scripts/weigths/"
        self.actor_model.save_weights(model_path+model_name + "_actor.h5")
        self.critic_model.save_weights(model_path+model_name + "_critic.h5")

        self.target_actor.save_weights(model_path+model_name + "_target_actor.h5")
        self.target_critic.save_weights(model_path+model_name + "_target_critic.h5")

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self,target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))


    def get_actor(self, model_weight):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.n_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs_ = [0,0]
        outputs_[0] = self.action_lower_bound[0] + ((self.action_upper_bound[0]-self.action_lower_bound[0])/2)
        outputs_[1] = self.action_lower_bound[1] + ((self.action_upper_bound[1]-self.action_lower_bound[1])/2)
        print ("outputs: " + str(outputs))
        print ("outputs_: " + str(outputs_))

        #print("Starting operations...")
        #print("test1: " + str(outputs_[0]*(outputs_[0]-1.0)))
        #print("test2: " + str(outputs-[-1.0,-1.0]))

        outputs = outputs_*(outputs-[-1.0,-1.0])

        #outputs[0] = outputs_[0]*(outputs_[0]-1.0)
        #outputs[1] = outputs_[1]*(outputs_[1]-1.0)

        model = tf.keras.Model(inputs, outputs)

        if(self.use_model):
            model.load_weights(model_weight)
        return model


    def get_critic(self,model_weight):
        # State as input
        state_input = layers.Input(shape=(self.n_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.n_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        if(self.use_model):
            model.load_weights(model_weight)

        return model
    def policy(self,state,add_noise):

        #normalize the array
        # print(state)
        # state__ = np.array(state)
        # norm = np.linalg.norm(state__)
        # state_norm = state/norm
        # print(state_norm)
        sampled_actions = tf.squeeze(self.actor_model(state))
        print("sampled_actions(Before noise)")
        print(str(sampled_actions.numpy()))
        # noise = self.noise_object()
        # Adding noise to action
        #print("----------------------")
        #print("Action")
        #print(str(sampled_actions))
        # print(str(noise))
        # print(str(sampled_actions.numpy()))
        # 2 is because tahn activation function return from -1 to 1
        # scaled_action = self.action_lower_bound + ((self.action_upper_bound-self.action_lower_bound)/2)*(sampled_actions.numpy()-[-1,-1])
        # print("prescaled action: "+ str(sampled_actions.numpy()))
        # print("scaled action " + str(scaled_action))
        #
        #try guassian noise
        noise_guas = tf.random.normal(shape = [self.n_actions], mean = [0,0], stddev = self.std_dev)
        if(add_noise == True):
            print("Adding noise")
            sampled_actions = sampled_actions.numpy() + noise_guas
            print(str(noise_guas))
        else:
            sampled_actions = sampled_actions.numpy()


        # sampled_actions = scaled_action + noise
        # print(str(sampled_actions))

        #print("Without clip action")
        #print(str(sampled_actions))

        #print("----------------------")

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.action_lower_bound, self.action_upper_bound)
        print("Final legal_action")
        print(str(legal_action))
        #print("clip action")
        #print(str(sampled_actions))

        return [np.squeeze(legal_action)]



# # testing
# problem = "Pendulum-v0"
# env = gym.make(problem)
#
# num_states = env.observation_space.shape[0]
# #print("Size of State Space ->  {}".format(num_states))
# #print(type(env.observation_space))
# num_actions = env.action_space.shape[0]
# #print("Size of Action Space ->  {}".format(num_actions))
# #print(type(env.action_space))
#
# upper_bound = env.action_space.high[0]
# lower_bound = env.action_space.low[0]
#
# #print("Max Value of Action ->  {}".format(upper_bound))
# #print("Min Value of Action ->  {}".format(lower_bound))
#
# std_dev = 0.2
# # Learning rate for actor-critic models
# critic_lr = 0.002
# actor_lr = 0.001
# buffer_capacity = 50000
# batch_size = 64
#
# total_episodes = 3
# # Discount factor for future rewards
# gamma = 0.99
# # Used to update target networks
# tau = 0.005
# # To store reward history of each episode
# ep_reward_list = []
# # To store average reward history of last few episodes
# avg_reward_list = []
#
# #initialing ddpg object
# ddpg_object = DDPG(n_actions = num_actions, n_states = num_states, action_bounds = [lower_bound, upper_bound], noise_std_dev = std_dev, critic_lr = critic_lr,actor_lr = actor_lr, buffer_capacity = buffer_capacity, batch_size = batch_size, tau = tau, gamma = gamma)
#
#
# # Takes about 4 min to train
# for ep in range(total_episodes):
#
#     prev_state = env.reset()
#     episodic_reward = 0
#     while True:
#         # Uncomment this to see the Actor in action
#         # But not in a python notebook.
#         #env.render()
#         #render()
#
#         tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
#
#         action = ddpg_object.policy(tf_prev_state)
#         # Recieve state and reward from environment.
#         state, reward, done, info = env.step(action)
#
#         ddpg_object.buffer.record((prev_state, action, reward, state))
#         episodic_reward += reward
#
#         ddpg_object.buffer.learn(ddpg_object.actor_model, ddpg_object.critic_model, ddpg_object.target_actor, ddpg_object.target_critic, ddpg_object.actor_optimizer, ddpg_object.critic_optimizer)
#         ddpg_object.update_target(ddpg_object.target_actor.variables, ddpg_object.actor_model.variables)
#         ddpg_object.update_target(ddpg_object.target_critic.variables, ddpg_object.critic_model.variables)
#
#         # End this episode when `done` is True
#         if done:
#             break
#
#         prev_state = state
#
#     ep_reward_list.append(episodic_reward)
#
#     # Mean of last 40 episodes
#     avg_reward = np.mean(ep_reward_list[-40:])
#     #print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
#     avg_reward_list.append(avg_reward)
# # Plotting graph
# # Episodes versus Avg. Rewards
# plt.plot(avg_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Avg. Epsiodic Reward")
# plt.show()
#
# # Save the weights
# ddpg_object.save_model_weigth("pendulum")
