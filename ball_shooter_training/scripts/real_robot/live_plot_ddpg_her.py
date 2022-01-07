#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
#import gym

rewards_key = 'episode_rewards'

class LivePlot(object):
    def __init__(self, outdir, data_key=rewards_key, line_color='blue'):
        self.outdir = outdir
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        # plt.xlabel("Episodes")
        # plt.ylabel(data_key)
        # fig = plt.gcf().canvas.set_window_title('simulation_graph')
        fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        fig.suptitle('simulation_graph')
        self.ax1.set_title("Reward")
        self.ax1.set(xlabel='episodes', ylabel='reward')
        self.ax2.set_title("Sucess rate")
        self.ax2.set(xlabel='episodes', ylabel='succes rate')

    def plot(self, reward_list, success_rate_list):
        self.ax1.plot(reward_list,color=self.line_color)
        self.ax2.plot(success_rate_list,color=self.line_color)

        # pause so matplotlib will display
        # may want to figure out matplotlib animation or use a different library in the future
        plt.pause(0.000001)
