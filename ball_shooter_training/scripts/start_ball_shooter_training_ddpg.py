#!/usr/bin/env python
############################################################
#####    Stevedan Ogochukwu Omodolor, November 2021    #####
#####Implementation of the reinforcement learning       ####
#####                                                   ####
############################################################
import ddpg
from gym import wrappers
import gym
import numpy
import numpy as np
import time
from functools import reduce
import os
os.getcwd()

#Ros packages required
import rospy
import rospkg
from live_plot import LivePlot
from std_msgs.msg import Float64
#inport our training environment
import ball_shooter_env_ddpg
import tensorflow as tf
import matplotlib.pyplot as plt
import random



if __name__ == '__main__':

    rospy.init_node('ballShooter_gym', anonymous=True, log_level=rospy.INFO)
    #create the gym environment
    env = gym.make("BallShooterEnvDdpg-v0")
    rospy.loginfo("Gym environment done")
    reward_pub = rospy.Publisher('/ball_shooter/reward', Float64, queue_size=1)
    episode_reward_pub = rospy.Publisher('/ball_shooter/episode_reward', Float64, queue_size=1)
    #setting the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ball_shooter_training')


    #setting the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ball_shooter_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")
    #plot
    plotter = LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/ball_shooter/alpha")
    Epsilon = rospy.get_param("/ball_shooter/epsilon")
    Gamma = rospy.get_param("/ball_shooter/gamma")
    epsilon_discount = rospy.get_param("/ball_shooter/epsilon_discount")
    nepisodes = rospy.get_param("/ball_shooter/nepisodes")
    nsteps = rospy.get_param("/ball_shooter/nsteps")
    n_actions_ddpg = rospy.get_param("/ball_shooter/n_actions_ddpg")
    n_states_ddpg = rospy.get_param("/ball_shooter/n_states_ddpg")
    max_pixel_x = rospy.get_param("/ball_shooter/max_pixel_x")
    min_pixel_x = rospy.get_param("/ball_shooter/min_pixel_x")
    max_pixel_y = rospy.get_param("/ball_shooter/max_pixel_y")
    min_pixel_y = rospy.get_param("/ball_shooter/min_pixel_y")
    upper_bound = env.action_space.high
    lower_bound = env.action_space.low

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    std_dev = [4, 10]
    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001
    buffer_capacity = 50000
    batch_size = 64
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    # #initialing ddpg object
    ddpg_object = ddpg.DDPG(n_actions = n_actions_ddpg, n_states = n_states_ddpg, lower_bound = lower_bound, upper_bound = upper_bound, noise_std_dev = std_dev, critic_lr = critic_lr,actor_lr = actor_lr, buffer_capacity = buffer_capacity, batch_size = batch_size, tau = tau, gamma = gamma)
    #
     # Initialises the algorithm that we are going to use for learning
    start_time = time.time()

    highest_reward = 0

    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01
    epsilon = 1

    # #start the main training loop:
    for x in range(nepisodes):
        rospy.loginfo("###################### START EPISODE=>" + str(x))
    #     cumulated_reward = 0
    #     cumulated_reward_msg = Float64()
        episode_reward_msg = Float64()
        step = 0
        done = False
        cumulated_reward = 0

    #
        #initialize the environment
        prev_state = env.reset()
        #mprove coce, nromalize in one area
        middle = (n_states_ddpg/2)
        prev_state_norm_list = list(prev_state)
        for t in range(n_states_ddpg):
            if(t <(middle)):
                prev_state_norm_list[t] = (prev_state_norm_list[t]-min_pixel_x)/(max_pixel_x-min_pixel_x)
            else:
                prev_state_norm_list[t] = (prev_state_norm_list[t]-min_pixel_y)/(max_pixel_y-min_pixel_y)
        prev_state = tuple(prev_state_norm_list)


        for i in range(nsteps):

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            exp_exp_tradeoff = random.uniform(0,1)
            if(exp_exp_tradeoff>epsilon):
                action = ddpg_object.policy(tf_prev_state, add_noise = False)
            else:
                action = ddpg_object.policy(tf_prev_state, add_noise = True)


            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)
            #normalize state vectors
            # middle = (n_states_ddpg/2)
            state_norm_list = list(state)
            for p in range(n_states_ddpg):
                if(p <(middle)):
                    state_norm_list[p] = (state_norm_list[p]-min_pixel_x)/(max_pixel_x-min_pixel_x)
                else:
                    state_norm_list[p] = (state_norm_list[p]-min_pixel_y)/(max_pixel_y-min_pixel_y)
            state_norm_tuple = tuple(state_norm_list)

            #previous state = current state
            ddpg_object.buffer.record(state_norm_tuple, action, reward, state_norm_tuple)
            #episode_reward_msg += reward
            ddpg_object.buffer.learn(ddpg_object.actor_model, ddpg_object.critic_model, ddpg_object.target_actor, ddpg_object.target_critic, ddpg_object.actor_optimizer, ddpg_object.critic_optimizer)
            ddpg_object.update_target(ddpg_object.target_actor.variables, ddpg_object.actor_model.variables)
            ddpg_object.update_target(ddpg_object.target_critic.variables, ddpg_object.critic_model.variables)

            # End this episode when `done` is True


            # choose a random actions
    #         action = qlearn.chooseAction(state) # todo how we choose action
    #         #if qlearn
    #         # #execute ction in the environment and get feedback
    #         rospy.loginfo("###################### Start Step...["+str(i)+"]")
            rospy.loginfo("Action to Perform >> "+str(action))
    #         state, reward,done,info = env.step(action)
            rospy.loginfo("Reward ==> " + str(reward))
            rospy.loginfo("state ==> " + str(state_norm_tuple))
            rospy.loginfo("done ==> " + str(done))
    #
            cumulated_reward=reward

            prev_state = state
    #         # if(highest_reward < cumulated_reward):
    #         #     highest_reward = culmulated_reward
    #         #
    #         # #make the algorithm learn here
    #         rospy.logwarn("############### state we were=>" + str(state))
    #         rospy.logwarn("############### action that we took=>" + str(action))
    #         rospy.logwarn("############### reward that action gave=>" + str(reward))
    #         qlearn.learn(state, action, reward, state)
    #
    #         # #publsh culmulated reward
            # culmulated_reward_msg.data = cumulated_reward
            # reward_pub.publish(reward)
    #         #
            if(done):
                break
            # rospy.loginfo("###################### END Step...["+str(i)+"]")
        # reduce epsilon
        epsilon = min_epsilon +(max_epsilon -min_epsilon)*np.exp(-decay_rate*x)
        print("epsilon: " + str(epsilon))
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
    #     # alpha = 0
    #     # gamma = 0
    #     # epsilon = 0
        episode_reward_msg.data = cumulated_reward
        episode_reward_pub.publish(episode_reward_msg)
        rospy.loginfo("Episode * {} * Avg Reward is ==> {}".format(x, cumulated_reward))
        avg_reward_list.append(cumulated_reward)

    #     rospy.loginfo( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))
        plotter.plot(env)
    # rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))
    #
    l = last_time_steps.tolist()
    l.sort()
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
    #
    # #print("Parameters: a="+str)
    # # rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # Save the weights
    model_path = "/home/stevedan/RL/RL_ws/src/ball_shooter/ball_shooter_training/scripts/weigths/ballShooter_actor.h5"
    ddpg_object.save_model_weigth(model_path,"ballShooter_fixed_bin_fixed_pan_joint")
    # ddpg_object.actor_model.save_weights("./weigths/ballShooter_actor.h5")
    # ddpg_object.critic_model.save_weights("./weigths/ballShooter_critic.h5")
    #
    # ddpg_object.target_actor.save_weights("./weigths/ballShooter_target_actor.h5")
    # ddpg_object.target_critic.save_weights("./weigths/ballShooter_target_critic.h5")
    env.close()
