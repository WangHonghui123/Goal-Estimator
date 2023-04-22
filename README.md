# Goal-Estimator
Goal-conditioned deep learning for trajectory prediction is to predict the goal position in advance and then uses the estimated goal positions to forecast the trajectory. Therefore, the accuracy of goal estimation is essential.

The goal estimator is from this paper:
```bibtex
@inproceedings{he2021where,
  title={Where are you heading? Dynamic Trajectory Prediction with Expert Goal Examples},
  author={He, Zhao and Richard P. Wildes},
  booktitle = {Proceedings of the International Conference on Computer Vision (ICCV)},
  month = {Oct.},
  year={2021}
}
```
Note that the result is different from the paper shown above.

The histogram of this goal estimator on ETH/UCY dataset is shown as follows:

<img src="goal_estimated_result/ETH_UCY (Normalized)/goal_estimated_histogram_eth.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Non_Normalized)/goal_estimated_histogram_eth.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Normalized)/goal_estimated_histogram_hotel.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Non_Normalized)/goal_estimated_histogram_hotel.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Normalized)/goal_estimated_histogram_univ.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Non_Normalized)/goal_estimated_histogram_univ.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Normalized)/goal_estimated_histogram_zara1.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Non_Normalized)/goal_estimated_histogram_zara1.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Normalized)/goal_estimated_histogram_zara2.png" style="display: inline; border-width: 0px;" width=400px></img>
<img src="goal_estimated_result/ETH_UCY (Non_Normalized)/goal_estimated_histogram_zara2.png" style="display: inline; border-width: 0px;" width=400px></img>


The Final Displacement Error is shown as follows:

Note: FDE means L2 norm between predicted goals and ground truth goals.

| **Dataset**                           | **FDE(Normalization)** | **FDE(Non-Normalization)**  |
| --------------------------------- | --------- |-----------------|
| **ETH**                     | 0.6516   |3.1039|
| **HOTEL**                     | 0.1474   |8.8868|
| **ZARA1**                     | 0.3106   |0.9238|
| **ZARA2**                     |  0.2555  |2.9344|
| **UNIV**                     | 0.6470   |2.1889|
