import pickle
import numpy as np
import matplotlib.pyplot as plt

open_directory = "./goal_estimated_result/ETH_UCY (Normalized)/"
dataset = ['univ', 'eth', 'zara1', 'zara2', 'hotel']
save_image_directory = "./goal_estimated_result/ETH_UCY (Normalized)/"

for dataname in dataset:
    open_error_file = open_directory + "goal_estimated_error_" + dataname + "_2.pkl"
    open_estimation_file = open_directory + "goal_estimated_" + dataname + "_2.pkl"
    error_file = open(open_error_file,'rb')
    error = pickle.load(error_file)
    error = np.concatenate(error['Estimated_Goal_Error'])
    aver_error = sum(error) / len(error)
    pedestrian = np.arange(0,len(error))
    plt.hist(error, density=True)
    plt.xlabel('L2-norm error')
    plt.ylabel('The frequency of L2-norm error')
    plt.title(f'The goal estimation on {dataname} with normalization\n Average L2-norm loss {aver_error}')
    plt.savefig(save_image_directory + f'goal_estimated_histogram_{dataname}.png')
    plt.show()
