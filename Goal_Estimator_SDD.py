import argparse
import torch
from soft_dtw_cuda import *
import yaml
from utils_SDD import *


def rotate_pc(coords, alpha):
    alpha = alpha * np.pi / 180
    M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return M @ coords

def expert_find(data, data_ori, expert_set, expert_ori, angles=None):
    global args
    """
    data: processed test dataset (pedestrian numbers, frame, position)
    data_ori : pre-processed train dataset [list: (pedestrian numbers, frame, position)]
    expert_set : processed train dataset (all pedestrians number stack, frame, position) (do it)
    expert_ori : pre-processed train dataset (all pedestrians number stack, frame, position) (do it)
    angles : used for data argumentation
    """

    all_min_end = [] # Store the minimum distance between all predicted goals and truth goals
    rest_diff = [] # Store the predicted goal with the minimum distance from the corresponding truth goal
    ceriterion = SoftDTW(
        use_cuda=False,
        gamma=2.0,
        normalize=True,
    )

    mse = torch.nn.MSELoss()

    num_of_trajs = data.shape[0] #The number of all pedestrians in processed test dataset
    print("Total number of searched data {} during the goal estimation".format(num_of_trajs))
    """Pre-process to velocity and accer"""

    #calculating the velocity and accerlation of the processed test dataset trajectory (pedestrian number, frame, position) using finite difference
    gradient_eff = 0.6
    traj_v = np.gradient(np.transpose(data, (0, 2, 1)), gradient_eff, axis=-1)
    traj_a = np.gradient(traj_v, gradient_eff, axis=-1)
    traj_v = torch.from_numpy(traj_v).permute(0, 2, 1).cuda()
    traj_a = torch.from_numpy(traj_a).permute(0, 2, 1).cuda()

    # TODO: apply random rotation on expert_set and expert_ori here
    extra_data = []
    extra_ori = []
    if angles is not None:
        for ang in angles:
            expert_copy = np.copy(expert_set)
            expert_ori_copy = np.copy(expert_ori)
            B, T, C = expert_copy.shape
            expert_copy = expert_copy.reshape(B * T, C).transpose()
            expert_ori_copy = expert_ori_copy.reshape(B * T, C).transpose()

            expert_copy = rotate_pc(expert_copy, ang).transpose()
            expert_ori_copy = rotate_pc(expert_ori_copy, ang).transpose()
            extra_data.append(expert_copy.reshape(B, T, C))
            extra_ori.append(expert_ori_copy.reshape(B, T, C))

    expert_set = np.concatenate(extra_data, 0) #used for next goal estimation
    expert_ori = np.concatenate(extra_ori, 0) #NO USE

    expert_traj_v = np.gradient(
        np.transpose(expert_set, (0, 2, 1)), gradient_eff, axis=-1
    )
    expert_traj_a = np.gradient(expert_traj_v, gradient_eff, axis=-1) #NO USE
    expert_traj_v = torch.from_numpy(expert_traj_v).permute(0, 2, 1).cuda()
    expert_traj_a = torch.from_numpy(expert_traj_a).permute(0, 2, 1).cuda() #NO USE

    expert_set = torch.from_numpy(expert_set).cuda()
    expert_ori = torch.from_numpy(expert_ori).cuda() #NO USE
    data = torch.DoubleTensor(data).to(device)
    data_ori = torch.DoubleTensor(data_ori).to(device).squeeze() #NO USE

    """
        For random few shot ablation study 
    """
    # random_split_ratio = 0.9
    # expert_size = expert_traj_v.shape[0]
    # print(int(expert_size * random_split_ratio))
    # indice = random.sample(range(expert_size), int(expert_size * random_split_ratio))
    # indice = torch.tensor(indice)
    # print(len(set(indice)))
    # expert_traj_v = expert_traj_v[indice]
    # expert_set = expert_set[indice]
    # print(expert_traj_v.shape)

    # t0 = time.time()
    for i in range(num_of_trajs):
        print(f'Deal with goal estimation of pedestrian No. {i+1} / {num_of_trajs}')
        tmp_traj_v = traj_v[i, :8].unsqueeze(0) #The velocity of observed parts (observed trajectory) in processed test dataset
        tmp_traj_abs = data[i, :8].unsqueeze(0) #NO USE!

        expert_num = expert_traj_v.shape[0]

        tmp_traj_v = tmp_traj_v.repeat(expert_num, 1, 1)
        tmp_traj_abs = tmp_traj_abs.repeat(expert_num, 1, 1) #NO USE!

        # DTW algorithm is used to calculate the distance between
        # the velocity of observed parts (observed trajectory) in processed test dataset (tmp_traj_v) and
        # the velocity of observed parts (observed trajectory) in processed train dataset (expert_traj_v[:, :8])
        loss = ceriterion(tmp_traj_v, expert_traj_v[:, :8])

        if args.eval_opt == 1:
            """Opt1: for dtw matching only"""
            #calculate the 20 lowest loss value and its index
            #min_k: the 20 lowest loss
            #min_k_indices: the corresponding index of the 20 lowest loss
            min_k, min_k_indices = torch.topk(loss, 20, largest=False)

        elif args.eval_opt == 2:
            """Opt2: for dtw matching + clustering matching"""
            #calculate the 65 lowest loss value and its index
            #min_k: the 65 lowest loss
            #min_k_indices: the corresponding index of the 65 lowest loss
            min_k, min_k_indices = torch.topk(loss, 65, largest=False)
            #retrieved_expert: the ground true goal positions with lowest 65 loss in processed train dataset
            retrieved_expert = expert_set[min_k_indices][:, -1]
            from sklearn.cluster import KMeans
            #cluster 20 class for these ground true goal positions with lowest 65 loss in processed train dataset
            kmeans = KMeans(n_clusters=20, random_state=0).fit(
                retrieved_expert.cpu().numpy()
            )

        iter_target = min_k_indices

        min_k_end = [] # Store the distance between predicted goals and true goals
        end_point_appr = [] # Store predicted goals

        """Back to indexing in real coords domain"""
        if args.eval_opt == 1:
            for k in iter_target:
                test_end = data[i, -1] #the ground truth goal position in the processed test dataset
                exp_end = expert_set[k, -1] #the ground true goal positions in processed train dataset
                #calculate the difference between every pedestrian's ground truth goal position in processed test dataset
                #and every pedestrian's ground truth goal position in processed train dataset
                min_k_end.append(torch.norm(test_end - exp_end, 2))
                end_point_appr.append(exp_end)

            # calculate the minimum difference between every pedestrian's ground truth goal position in processed test dataset
            # and every pedestrian's ground truth goal position in processed train dataset
            # representing which ground truth goal position in processed train dataset
            # ground truth goal position in processed test dataset is more similar
            all_min_end.append(min(min_k_end))
            # predicted goal is the ground truth goal position in processed train dataset that is most similar to
            # ground truth goal position in processed test dataset
            rest_diff.append(end_point_appr[min_k_end.index(min(min_k_end))])

            # print('-----------------------Using DTW matching only---------------------')
            # print(f'The error of No. {i+1} pedestrian between its ground truth goal position '
            #       f'and its predicted goal position is {min(min_k_end)}')
            # print('-----------------------Using DTW matching only---------------------')

        else:
            for k in kmeans.cluster_centers_:
                test_end = data[i, -1] #the ground truth foal position in the processed test dataset
                exp_end = torch.from_numpy(k).cuda() #the cluster center

                # min_k_end: calculate the difference between every pedestrian's ground truth goal position
                # in the processed test dataset and all the cluster center (20 in total)
                min_k_end.append(torch.norm(test_end - exp_end, 2))
                end_point_appr.append(exp_end)

            # calculate the minimum distance between every pedestrian's ground truth goal position
            # in the processed test dataset and all the cluster center (20 in total)
            # representing which clusters the ground truth goal position in the processed test dataset belongs to
            all_min_end.append(min(min_k_end))
            # predicted goal is the cluster center of which clusters every pedestrian's ground truth goal position
            # in the processed test belongs to
            rest_diff.append(end_point_appr[min_k_end.index(min(min_k_end))])

            # print('-----------------------Using DTW matching and clustering matching---------------------')
            # print(f'The error of No. {i + 1} pedestrian between its ground truth goal position '
            #       f'and its predicted goal position is {min(min_k_end)}')
            # print('-----------------------Using DTW matching and clustering matching---------------------')

    print('Goal estimation is done................................................')
    return all_min_end, rest_diff


def sdd_goal_estimator(test_dataset, train_dataset):
    estimated_goal_result = {'Predicted_Goal': [], 'True_Goal': []}
    estimated_goal_error = {'Estimated_Goal_Error': []}

    for i, (traj, mask, initial_pos) in enumerate(
        zip(
            test_dataset.trajectory_batches,
            test_dataset.mask_batches,
            test_dataset.initial_pos_batches,
        )
    ):
        # processed test dataset (pedestrian numbers, frame, position)
        traj_np = np.copy(traj) # processed test trajectory
        ground_truth_goal = traj_np[:,-1]  # ground truth goal: [num_of_objs, 2]
        estimated_goal_result['True_Goal'].append(ground_truth_goal)

        # pre-processed train dataset (all pedestrians number stack, frame, position)
        expert_ori = train_dataset.trajectory_ori
        expert_ori_list = [x for x in expert_ori]
        expert_ori = np.concatenate(expert_ori_list, 0)
        # processed train dataset (all pedestrians number stack, frame, position)
        expert_traj = train_dataset.trajectory_batches
        expert_traj_list = [x for x in expert_traj]
        expert_traj = np.concatenate(expert_traj_list, 0)

        angles = [0]
        end_error, rst = expert_find(
                traj_np,
                test_dataset.trajectory_ori,
                expert_traj,
                expert_ori,
                angles,
        )
        rst = torch.stack(rst)  # All selected predicted goals: [num_of_objs, 2]
        estimated_goal_result['Predicted_Goal'].append(rst.numpy())

        """Find the goal retrieval that is too wrong, i.e. > 100 pixels, do not trust this result anymore;
        """
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
    parser.add_argument(
        "--eval_opt",
        type=int,
        default=2,
        help="specify ways to search: 1 for dtw; 2 for dtw + clustering",
    )
    args = parser.parse_args()

    '''
    For torch precision and device (CPU/GPU)
    '''

    # torch precision and device setting for CPU/GPU
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print('Torch precision and devide setting:.........................................')
    print(device)

    '''
    For hyper parameters
    '''

    # load running-related parameters
    with open("./config/" + args.config_filename, "r") as file:
        try:
            hyper_params = yaml.load(file, Loader=yaml.FullLoader)
        except:
            hyper_params = yaml.load(file)
    file.close()
    print('The model and running-related parameters...................................')
    print(hyper_params)

    '''
    For dataset (train set and test set)
    '''

    # load TRAIN dataset
    train_dataset = SocialDataset(
        set_name="train",
        b_size=hyper_params["train_b_size"],
        t_tresh=hyper_params["time_thresh"],
        d_tresh=hyper_params["dist_thresh"],
        verbose=args.verbose,
    )
    # load TEST dataset
    test_dataset = SocialDataset(
        set_name="test",
        b_size=hyper_params["test_b_size"],
        t_tresh=hyper_params["time_thresh"],
        d_tresh=hyper_params["dist_thresh"],
        verbose=args.verbose,
    )

    # TRAIN dataset processing (train_dataset is going to be changed)
    for traj in train_dataset.trajectory_batches:
        traj -= traj[:, :1, :]
        traj *= 0.2
    # TEST dataset processing (test_dataset is going to be changed)
    for traj in test_dataset.trajectory_batches:
        traj -= traj[:, :1, :]
        traj *= 0.2


    '''
    For save
    '''
    save_directory = "./goal_estimated_result/SDD/"
    #if there is not a directory under a specified route, it will be created.
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)


    '''
    For GOAL ESTIMATOR
    '''
    # goal_estimated: [num_of_objs, 2]:  All selected predicted goals
    # estimation_error: [num_of_objs, 1]: All estimated error (represented by l2 loss between all selected predicted goals and all corresponding truth goals)
    goal_estimated, estimation_error = sdd_goal_estimator(test_dataset, train_dataset)

    # goal_estimated_1 and goal_estimated_error_1 are using DTW matching only
    # goal_estimated_2 and goal_estimated_error_2 are using DTW matching + clustering matching
    print('Save Model...')
    with open(os.path.join(save_directory, f'goal_estimated_{args.eval_opt}.pkl'), 'wb') as f:
        pickle.dump(goal_estimated, f)

    print('Save error...')
    with open(os.path.join(save_directory, f'goal_estimated_error_{args.eval_opt}.pkl'), 'wb') as f:
        pickle.dump(estimation_error, f)







