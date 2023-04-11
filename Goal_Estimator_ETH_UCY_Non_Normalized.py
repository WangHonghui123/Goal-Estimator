import argparse
import torch
import numpy as np
from utils_ETH_UCY_Non_Normalized import *
from helper_expert import *
import pickle


class Data_Expert:
    def __init__(self, obs_traj_norm, velocity_obs, pred_traj_gt):
        self.obs_traj_norm = obs_traj_norm
        self.velocity_obs = velocity_obs
        self.pred_traj_gt = pred_traj_gt

def test(dataset_name):

    '''
    For dataset (train, val and test set)
    '''

    # dataset_name = "univ"
    # dataset_name = "eth"
    # dataset_name = "zara1"
    # dataset_name = "zara2"
    # dataset_name = "hotel"
    dataset_path = "./datasets/" + dataset_name + "/"

    obs_seq_len = 8
    pred_seq_len = 12
    grad_eff = 0.4

    dset_train = TrajectoryDataset(
        dataset_path + "train/",
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,
        norm_lap_matr=True,
        grad_eff=grad_eff,
    )

    dset_val = TrajectoryDataset(
        dataset_path + "val/",
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,
        norm_lap_matr=True,
        grad_eff=grad_eff,
    )

    dset_test = TrajectoryDataset(
        dataset_path + "test/",
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,
        norm_lap_matr=True,
        grad_eff=grad_eff,
    )

    loader_test = DataLoader(
        dset_test,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=1,
    )
    '''
    For log and save result
    '''
    estimated_goal_result = {'Predicted_Goal': [], 'True_Goal': []}
    estimated_goal_error = {'Estimated_Goal_Error': []}

    step = 0

    for batch in loader_test:
        step += 1
        # Get data
        # batch = [tensor.cuda() for tensor in batch]
        batch = [tensor for tensor in batch]
        '''
        Load the batch data
        '''
        (
            obs_traj_norm,
            obs_traj,
            obs_traj_rel,
            pred_traj_gt,
            pred_traj_gt_rel,
            V_obs,
            A_obs,
            V_tr,
            A_tr,
            inp_mask,
            out_mask,
            velocity_obs,
            velocity_pred,
            acc_obs,
            acc_pred,
            seq_start,
        ) = batch

        """
        Perform the experties matching here
        """
        num_of_objs = int(sum(inp_mask[0, 0]))
        data = Data_Expert(obs_traj_norm, velocity_obs, pred_traj_gt)
        end_error, rst = expert_find(data, num_of_objs, dset_train, dset_val, step, gamma=1.0)

        '''
        Store the ground true goal and predicted goal
        '''
        rst = torch.stack(rst)  # [num_of_objs, 2]
        estimated_goal_result['Predicted_Goal'].append(rst.numpy())
        ground_truth_goal = pred_traj_gt[0,-1,:num_of_objs]  # ground truth goal: [num_of_objs, 2]
        estimated_goal_result['True_Goal'].append(ground_truth_goal.numpy())

        end_error = torch.stack(end_error)  # The distance between all selected predicted goals and corresponding truth goals
        estimated_goal_error['Estimated_Goal_Error'].append(end_error.numpy())

    return estimated_goal_result, estimated_goal_error


if __name__ == '__main__':
    # set command line parsing module
    parser = argparse.ArgumentParser(description="GoalExample")
    parser.add_argument("--num_workers", "-nw", type=int, default=0)
    parser.add_argument("--gpu_index", "-gi", type=int, default=0)
    parser.add_argument("--config_filename", "-cfn", type=str, default="optimal.yaml")
    parser.add_argument("--save_file", "-sf", type=str, default="PECNET_social_model.pt")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--input_feat", type=int, default=2, help="learning rate")
    parser.add_argument("--output_feat", type=int, default=128, help="learning rate")
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoint_sdd_abs2", help="learning rate"
    )
    # For ETH/UCY dataset, option 2 only
    parser.add_argument(
        "--eval_opt",
        type=int,
        default=2,
        help="specify ways to search: 1 for dtw; 2 for dtw + clustering",
    )
    args = parser.parse_args()

    '''
    For save
    '''
    save_directory = "./goal_estimated_result/ETH_UCY (Non_Normalized)/"
    #if there is not a directory under a specified route, it will be created.
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    '''
    For Goal Estimator
    '''
    dataset = ['univ','eth','zara1','zara2','hotel']
    for dataname in dataset:
        print(f'Calculating the goal estimation of {dataname}')
        goal_estimated, estimation_error = test(dataname)
        # goal_estimated_2 and goal_estimated_error_2 are using DTW matching + clustering matching
        print('Save Model...')
        with open(os.path.join(save_directory, f'goal_estimated_{dataname}_{args.eval_opt}.pkl'), 'wb') as f:
            pickle.dump(goal_estimated, f)

        print('Save error...')
        with open(os.path.join(save_directory, f'goal_estimated_error_{dataname}_{args.eval_opt}.pkl'), 'wb') as f:
            pickle.dump(estimation_error, f)






