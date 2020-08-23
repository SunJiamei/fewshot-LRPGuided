# fewshot-LRPGuided
This repo provides the LRP-guided few-shot classification models introduced in [Explanation-Guided Training for Cross-Domain Few-Shot Classification](https://arxiv.org/abs/2007.08790)

The implementation of explanation-guided models are based on the [fewshot-CAN](https://github.com/blue-blue272/fewshot-CAN) and the [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot) code.
We provide the LRP-guided CAN and GNN few shot classificatin models.

To train the two models, please update the [net.py](https://github.com/blue-blue272/fewshot-CAN/blob/master/torchFewShot/models/net.py) file of fewshot-CAN and the [gnnnet.py](https://github.com/hytseng0509/CrossDomainFewShot/blob/master/methods/gnnnet.py) of CrossDomainFewShot and follow the details of the two repos.
