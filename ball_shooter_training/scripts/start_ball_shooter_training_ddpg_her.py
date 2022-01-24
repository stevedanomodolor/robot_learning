#!/usr/bin/env python
############################################################
#####    Stevedan Ogochukwu Omodolor, November 2021    #####
#####Implementation of the reinforcement learning       ####
#####                                                   ####
############################################################
import ddpg_her
from gym import wrappers
import gym
import numpy
import numpy as np
import time
from functools import reduce
import os
import signal
import sys
import copy

os.getcwd()

#Ros packages required
import rospy
import rospkg
from live_plot_ddpg_her import LivePlot
from std_msgs.msg import Float64
#inport our training environment
import ball_shooter_env_ddpg_her
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from successRate import successRate
#
shut_down = False
def signal_handler(sig, frame):
    print('You pressed Ctrl+C, shutting down!')
    global shut_down
    shut_down = True

def normalize_state(min_pixel_y,max_pixel_y,max_pixel_x, min_pixel_x, list_array, num_s):
        # The state is organized as follows,, xxxxxyyyyyy
        middle = (num_s/2) # find this mid point
        new_array = []
        for t in range(n_states_ddpg):
            if(t <(middle)):
                new_array.append((list_array[t]-min_pixel_x)/(max_pixel_x-min_pixel_x))
            else:
                new_array.append((list_array[t]-min_pixel_y)/(max_pixel_y-min_pixel_y))
        return new_array



if __name__ == '__main__':

    rospy.init_node('ballShooter_gym', anonymous=True, log_level=rospy.INFO)
    #create the gym environment
    env = gym.make("BallShooterEnvDdpgHer-v0")
    rospy.loginfo("Gym environment done")
    reward_pub = rospy.Publisher('/ball_shooter/reward', Float64, queue_size=1)
    episode_reward_pub = rospy.Publisher('/ball_shooter/episode_reward', Float64, queue_size=1)
    success_rate_pub = rospy.Publisher('/ball_shooter/episode_success_rate', Float64, queue_size=1)
    #setting the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ball_shooter_training')


    #setting the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ball_shooter_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")
    signal.signal(signal.SIGINT, signal_handler)
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
    num_steps_to_average = rospy.get_param("/ball_shooter/num_steps_to_average")
    sucess_rate_reached = False
    max_succes_rate = rospy.get_param("/ball_shooter/max_succes_rate")
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
    # to store the succes rate history
    success_rate_list = []
    # #initialing ddpg object
    model_path_ = "/home/stevedan/RL/RL_ws/src/ball_shooter/ball_shooter_training/scripts/weigths/";
    model_actor_ = "ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3_actor.h5"
    model_critic_ = "ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3_critic.h5"
    model_actor_target_ = "ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3_target_actor.h5"
    model_critic_target_ = "ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3_target_critic.h5"
    ddpg_her_object = ddpg_her.DDPGHER(n_actions = n_actions_ddpg, n_states = n_states_ddpg,
                            lower_bound = lower_bound, upper_bound = upper_bound,
                            noise_std_dev = std_dev, critic_lr = critic_lr,actor_lr = actor_lr,
                            buffer_capacity = buffer_capacity, batch_size = batch_size, tau = tau,
                             gamma = gamma, use_model = True,model_path = model_path_, model_actor = model_actor_ ,
                            model_critic = model_critic_ , model_actor_target = model_actor_target_ ,
                            model_critic_target = model_critic_target_ )
    success_rate = successRate(num_steps_to_average)
        #
     # Initialises the algorithm that we are going to use for learning
    start_time = time.time()

    highest_reward = 0

    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01
    epsilon = 1

    # for the HER algorithm
    max_epsilon_her = 1.0
    min_epsilon_her = 0.01
    decay_rate_her = 0.03
    epsilon_her = 1
    x = 0

    # #start the main training loop:
    # for x in range(nepisodes):
    while True:
        x = x +1
        rospy.loginfo("###################### START EPISODE=>" + str(x))
        episode_reward_msg = Float64()
        success_rate_msg = Float64()
        step = 0
        done = False
        reward_ = 0
        #initialize the environment
        prev_state = env.reset()
        # print("previous satet outside1: " + str(prev_state))
        prev_state_norm = normalize_state(min_pixel_y,max_pixel_y,max_pixel_x, min_pixel_x, prev_state, n_states_ddpg)
        # print("previous satet outside2: " + str(prev_state_norm))


        for i in range(nsteps):

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state_norm), 0)
            # choose action
            exp_exp_tradeoff = random.uniform(0,1)
            if(exp_exp_tradeoff>epsilon):
                action = ddpg_her_object.policy(tf_prev_state, add_noise = False)
            else:
                action = ddpg_her_object.policy(tf_prev_state, add_noise = True)
            # Recieve state and reward from environment.
            # add noise in the action  only in the speed, inaccuracy in the real robot
            #copy value not reference
            # print("--------------------------------------------")
            # rospy.loginfo("Action to Perform >> "+str(action_real))
            # rospy.loginfo("Action to generated by RL >> "+str(action))
            # print("--------------------------------------------")
            #
            # mu = action_real[0][0]
            # sigma = 0.15*mu
            # print("---------------------------------------")
            # print("mu" + str(mu)+ " sigma "+str(sigma))
            # print("---------------------------------------")
            #
            # #vel = np.random.normal(mu, sigma,1)
            # #action_real[0][0] = vel[0]
            # print("vel real"+  str(vel[0]))
            state, reward, done, info = env.step(action)

            # add succes rate
            success_rate.put(reward)
            reward_ = reward
            # apply her using the same tradeoff technique: this is not used because it performs poorly
            if 0: #not reward == 1:
                exp_exp_tradeoff_her = random.uniform(0,1)
                if(exp_exp_tradeoff_her<epsilon_her):
                    rospy.loginfo("HER: relabeling goal")
                    # we put the bin at the location of the ball
                    ball_location = env.ball_shooter_object.get_ball_pose()
                    if(ball_location.pose.pose.position.x > 0.7):
                    # set the bin location to the new goal
                        env.ball_shooter_object.set_bin_location(ball_location.pose.pose.position.x, ball_location.pose.pose.position.y)
                        time.sleep(2)
                        # get newly labeled state
                        relabled_state = env.ball_shooter_object.get_state()
                        # relabel the data
                        reward = 1
                        state = relabled_state
                        # print("Inside ----- previous state: "+ str(prev_state))
                        # print("Inside -----  state: "+ str(state))

            state_norm = normalize_state(min_pixel_y,max_pixel_y,max_pixel_x, min_pixel_x, state, n_states_ddpg)
            #previous state = current state
            ddpg_her_object.buffer.record(state_norm, action, reward, state_norm)

            #episode_reward_msg += reward
            ddpg_her_object.buffer.learn(ddpg_her_object.actor_model, ddpg_her_object.critic_model, ddpg_her_object.target_actor, ddpg_her_object.target_critic, ddpg_her_object.actor_optimizer, ddpg_her_object.critic_optimizer)
            ddpg_her_object.update_target(ddpg_her_object.target_actor.variables, ddpg_her_object.actor_model.variables)
            ddpg_her_object.update_target(ddpg_her_object.target_critic.variables, ddpg_her_object.critic_model.variables)

            rospy.loginfo("Action to generated by RL >> "+str(action))
            rospy.loginfo("Reward ==> " + str(reward))
            rospy.loginfo("state ==> " + str(state))
            rospy.loginfo("done ==> " + str(done))
            if success_rate.get_average() > max_succes_rate:
                print("Sucess rate reached!! shutting down test")
                sucess_rate_reached = True




            # prev_state = state
            if(done):
                break
            # rospy.loginfo("###################### END Step...["+str(i)+"]")
        # reduce epsilon
        if (sucess_rate_reached==True) or (shut_down==True):
            break
        epsilon = min_epsilon +(max_epsilon -min_epsilon)*np.exp(-decay_rate*x)
        epsilon_her = min_epsilon_her +(max_epsilon_her -min_epsilon_her)*np.exp(-decay_rate_her*x)

        # print("epsilon: " + str(epsilon))
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        episode_reward_msg.data = reward_
        episode_reward_pub.publish(episode_reward_msg)
        success_rate_msg.data = success_rate.get_average()

        # rospy.loginfo("Episode * {} * Avg Reward is ==> {}".format(x, reward_))
        # rospy.loginfo("Episode * {} * succes rate is ==> {}".format(x, success_rate.get_average()))

        avg_reward_list.append(reward_)
        success_rate_list.append(success_rate.get_average()*100)

    #     rospy.loginfo( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))
        plotter.plot(avg_reward_list,success_rate_list)
    # rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))
    #
    l = last_time_steps.tolist()
    l.sort()
    # plt.plot(success_rate_list)
    # plt.xlabel("Episode")
    # plt.ylabel("Avg. Epsiodic Reward")
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Reward/succes rate')
    ax1.set_title("Reward")
    ax1.set(xlabel='episodes', ylabel='reward')
    ax1.plot(avg_reward_list)
    ax2.set_title("Sucess rate")
    ax2.set(xlabel='episodes', ylabel='succes rate')
    ax2.plot(success_rate_list)
    plt.show()

    #
    # #print("Parameters: a="+str)
    # # rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # Save the weights
    model_path = "/home/stevedan/RL/RL_ws/src/ball_shooter/ball_shooter_training/scripts/weigths/"
    ddpg_her_object.save_model_weigth(model_path,"ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3")
    # ddpg_her_object.actor_model.save_weights("./weigths/ballShooter_actor.h5")
    # ddpg_her_object.critic_model.save_weights("./weigths/ballShooter_critic.h5")
    #
    # ddpg_her_object.target_actor.save_weights("./weigths/ballShooter_target_actor.h5")
    # ddpg_her_object.target_critic.save_weights("./weigths/ballShooter_target_critic.h5")
    env.close()
    rospy.signal_shutdown("shutting down")
    sys.exit(0)
