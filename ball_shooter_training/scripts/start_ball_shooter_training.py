#!/usr/bin/env python
############################################################
#####    Stevedan Ogochukwu Omodolor, November 2021    #####
#####Implementation of the reinforcement learning       ####
#####                                                   ####
############################################################
import qlearn
from gym import wrappers
import gym
import numpy
import time
from functools import reduce

#Ros packages required
import rospy
import rospkg
from live_plot import LivePlot
from std_msgs.msg import Float64

#inport our training environment
import ball_shooter_env



if __name__ == '__main__':

    rospy.init_node('ballShooter_gym', anonymous=True, log_level=rospy.INFO)
    #create the gym environment
    env = gym.make("BallShooterEnv-v0")
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

    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
     # Initialises the algorithm that we are going to use for learning
    initial_epsilon = 0
    start_time = time.time()

    highest_reward = 0

    #start the main training loop:
    for x in range(nepisodes):
        rospy.loginfo("############### START EPISODE=>" + str(x))
        cumulated_reward = 0
        cumulated_reward_msg = Float64()
        episode_reward_msg = Float64()
        step = 0
        done = False

        #initialize the environment
        state = env.reset()
        for i in range(nsteps):
            # choose a random actions
            action = qlearn.chooseAction(state) # todo how we choose action
            #if qlearn
            # #execute ction in the environment and get feedback
            rospy.loginfo("###################### Start Step...["+str(i)+"]")
            # rospy.loginfo("Action to Perform >> "+str(action))
            state, reward,done,info = env.step(action)
            print("type: state " + str(type(state)))
            # rospy.loginfo("END Step...")
            # rospy.loginfo("Reward ==> " + str(reward))
            # rospy.loginfo("state ==> " + str(state))
            # rospy.loginfo("done ==> " + str(done))

            # cumulated_reward+=reward
            # if(highest_reward < cumulated_reward):
            #     highest_reward = culmulated_reward
            #
            # #make the algorithm learn here
            rospy.logwarn("############### state we were=>" + str(state))
            rospy.logwarn("############### action that we took=>" + str(action))
            rospy.logwarn("############### reward that action gave=>" + str(reward))
            qlearn.learn(state, action, reward, state)

            # #publsh culmulated reward
            # culmulated_reward_msg.data = cumulated_reward
            reward_pub.publish(reward)
            #
            if(done):
                break
            rospy.loginfo("###################### END Step...["+str(i)+"]")
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        # alpha = 0
        # gamma = 0
        # epsilon = 0
        episode_reward_msg.data = cumulated_reward
        episode_reward_pub.publish(episode_reward_msg)
        rospy.loginfo( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))
        plotter.plot(env)
    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    # rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))

    env.close()
