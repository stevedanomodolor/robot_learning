import numpy as np
class successRate:
    def __init__(self, num_to_average):
        self.buffer = np.zeros(num_to_average)
        self.num_to_average = num_to_average


    def put(self,reward):
        if int(reward) == 0:
            self.buffer = np.append(self.buffer, 1)
        else:
            self.buffer = np.append(self.buffer, 0)
    def get_average(self):
        #print("Succes rate buffer:  " + str(self.buffer[-self.num_to_average:]))
        return np.sum(self.buffer[-self.num_to_average:])/self.num_to_average
