
from ball_shooter_utils_rl_real_robot import BallShooterRLUtilsRealRobot
from live_plot_ddpg_her import LivePlot
from successRate import successRate
#from ddpg_her import DDPGHER
import ddpg_her

import numpy
import numpy as np
import time
from functools import reduce
import os
import signal
import sys
os.getcwd()

import tensorflow as tf
import matplotlib.pyplot as plt
import random

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
    outdir = '/home/guillem/RL_ws/src/robot_learning/ball_shooter_training/scripts'
    ball_shooter_object = BallShooterRLUtilsRealRobot()
    plotter = LivePlot(outdir)
    signal.signal(signal.SIGINT, signal_handler)


    last_time_steps = numpy.ndarray(0)
    # in config file
    Alpha = 0.1
    Epsilon = 0.9
    Gamma = 0.8
    epsilon_discount = 0.999
    nepisodes = 300
    nsteps = 1
    n_actions_ddpg = 2
    n_states_ddpg = 8
    max_pixel_x = 640
    min_pixel_x = 0
    max_pixel_y = 480
    min_pixel_y = 0
    max_succes_rate = 0.85


    vel_max = 8
    vel_min = 4
    pos_max = 30
    pos_min = -30

    upper_bound = [vel_max, pos_max] # canvia
    lower_bound = [vel_min, pos_min]

    std_dev = [6, 11]
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
    # To store average rewa rd history of last few episodes
    avg_reward_list = []
    # to store the succes rate history
    success_rate_list = []
    success_rate_reached = False
    # #initialing ddpg object
    model_path_ = "/home/gcornella/RL_ws/src/robot_learning/ball_shooter_training/scripts/weigths/"
    model_actor_ = "ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3_actor.h5"
    model_critic_ = "ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3_critic.h5"
    model_actor_target_ = "ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3_target_actor.h5"
    model_critic_target_ = "ballShooter_fixed_bin_fixed_pan_joint_ddpg_v3_target_critic.h5"

    ddpg_her_object = ddpg_her.DDPGHER(n_actions = n_actions_ddpg, n_states = n_states_ddpg,
                            lower_bound = lower_bound, upper_bound = upper_bound,
                            noise_std_dev = std_dev, critic_lr = critic_lr,actor_lr = actor_lr,
                            buffer_capacity = buffer_capacity, batch_size = batch_size, tau = tau,
                            gamma = gamma, use_model = True, model_path = model_path_, model_actor = model_actor_ ,
                            model_critic = model_critic_ , model_actor_target = model_actor_target_ ,
                            model_critic_target = model_critic_target_ )

    num_steps_to_average = 5
    success_rate = successRate(num_steps_to_average)
        #
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

    for x in range(nepisodes):
        print("###################### START EPISODE=>" + str(x))
        step = 0
        done = False
        reward_ = 0
        #initialize the environment
        prev_state = ball_shooter_object.get_state()
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
            # Perform the action
            ball_shooter_object.move_pan_tilt_launch_ball(action)
            state = prev_state
            input_correct = False
            while(not input_correct ):
                user_input = input("Select the reward: ")
                try:
                    reward = int(user_input)
                    print("This is a valid number!: ", reward)
                    input_correct = True
                    break
                except ValueError:
                    print("This is not a valid number. Try again :)")
                    input_correct = False

            # add succes rate
            success_rate.put(reward)
            # apply her using the same tradeoff technique
            # if 0: #not reward == 1:
            #     exp_exp_tradeoff_her = random.uniform(0,1)
            #     if(exp_exp_tradeoff_her<epsilon_her):
            #         rospy.loginfo("HER: relabeling goal")
            #         # we put the bin at the location of the ball
            #         ball_location = env.ball_shooter_object.get_ball_pose()
            #         if(ball_location.pose.pose.position.x > 0.7):
            #         # set the bin location to the new goal
            #             env.ball_shooter_object.set_bin_location(ball_location.pose.pose.position.x, ball_location.pose.pose.position.y)
            #             time.sleep(2)
            #             # get newly labeled state
            #             relabled_state = env.ball_shooter_object.get_state()
            #             # relabel the data
            #             reward = 1
            #             state = relabled_state
                        # print("Inside ----- previous state: "+ str(prev_state))
                        # print("Inside -----  state: "+ str(state))

            #previous state = current state
            state_norm = prev_state_norm
            ddpg_her_object.buffer.record(state_norm, action, reward, state_norm)

            #episode_reward_msg += reward
            ddpg_her_object.buffer.learn(ddpg_her_object.actor_model, ddpg_her_object.critic_model, ddpg_her_object.target_actor, ddpg_her_object.target_critic, ddpg_her_object.actor_optimizer, ddpg_her_object.critic_optimizer)
            ddpg_her_object.update_target(ddpg_her_object.target_actor.variables, ddpg_her_object.actor_model.variables)
            ddpg_her_object.update_target(ddpg_her_object.target_critic.variables, ddpg_her_object.critic_model.variables)

            print("Action to Perform >> "+str(action))
            print("Reward ==> " + str(reward))
            print("state ==> " + str(state))
            print("done ==> " + str(done))
            if success_rate.get_average() > max_succes_rate:
                print("Success rate reached!! shutting down test")
                success_rate_reached = True
        if (success_rate_reached==True) or (shut_down==True):
            break
        epsilon = min_epsilon +(max_epsilon -min_epsilon)*np.exp(-decay_rate*x)
        # epsilon_her = min_epsilon_her +(max_epsilon_her -min_epsilon_her)*np.exp(-decay_rate_her*x)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        avg_reward_list.append(reward)
        success_rate_list.append(success_rate.get_average())

        plotter.plot(avg_reward_list,success_rate_list)
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
    # Save the weights
    ddpg_her_object.save_model_weigth(model_path,"ballShooter_fixed_bin_fixed_pan_joint_ddpg_her")
    sys.exit(0)
