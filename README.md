# Goal-Estimator
For research use only

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
The histogram of this goal estimator on ETH/UCY dataset is shown as follows:

<img src="goal_estimated_result/ETH_UCY/goal_estimated_histogram_eth.png" style="display: inline; border-width: 0px;" width=500px></img>
<img src="goal_estimated_result/ETH_UCY/goal_estimated_histogram_hotel.png" style="display: inline; border-width: 0px;" width=500px></img>
<img src="goal_estimated_result/ETH_UCY/goal_estimated_histogram_univ.png" style="display: inline; border-width: 0px;" width=500px></img>
<img src="goal_estimated_result/ETH_UCY/goal_estimated_histogram_zara1.png" style="display: inline; border-width: 0px;" width=500px></img>
<img src="goal_estimated_result/ETH_UCY/goal_estimated_histogram_zara2.png" style="display: inline; border-width: 0px;" width=500px></img>

The Final Displacement Error is shown as follows:


| **Dataset**                           | **FDE(L2 norm between predicted goals and ground truth goals)** |
| --------------------------------- | ------- |
| **ETH**                     | 0.9735   |
| **HOTEL**                     | 0.4942   |
| **ZARA1**                     | 0.3402   |
| **ZARA2**                     |  0.2646  |
| **UNIV**                     | 0.6469   |
