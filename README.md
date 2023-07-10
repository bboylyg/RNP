# Reconstructive Neuron Pruning for Backdoor Defense

Code for ICML 2023 Paper ["Reconstructive Neuron Pruning for Backdoor Defense"](https://arxiv.org/pdf/2305.14876.pdf)

# Quick Start: RNP against BadNets Attack  
By default, we only use 500 defense data randomly sampled from the training set to perform the `unlearn-recover` process and optimize the pruning mask. To check the performance of RNP on a Badnets ResNet-18 network (i.e. 10% poisoning rata with ResNet-18 on CIFAR-10), you can directly run the command like:

```python
python main.py
```

# Experimental Results on BadNets Attack  

```python
[2023/07/09 22:21:00] - Namespace(alpha=0.2, arch='resnet18', backdoor_model_path='weights/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar', batch_size=128, clean_threshold=0.2, cuda=1, dataset='CIFAR10', log_root='logs/', mask_file=None, momentum=0.9, num_class=10, output_weight='weights/', pruning_by='threshold', pruning_max=0.9, pruning_step=0.05, ratio=0.01, recovering_epochs=20, recovering_lr=0.2, save_every=5, schedule=[10, 20], target_label=0, target_type='all2one', trig_h=3, trig_w=3, trigger_type='gridTrigger', unlearned_model_path=None, unlearning_epochs=20, unlearning_lr=0.01, weight_decay=0.0005)
[2023/07/09 22:21:00] - ----------- Data Initialization --------------
[2023/07/09 22:21:03] - ----------- Backdoor Model Initialization --------------
[2023/07/09 22:21:04] - Epoch 	 lr 	 Time 	 TrainLoss 	 TrainACC 	 PoisonLoss 	 PoisonACC 	 CleanLoss 	 CleanACC
[2023/07/09 22:21:04] - ----------- Model Unlearning --------------
[2023/07/09 22:21:15] - 0 	 0.010 	 11.2 	 0.1004 	 0.9780 	 0.0000 	 1.0000 	 0.2185 	 0.9342
[2023/07/09 22:21:26] - 1 	 0.010 	 10.8 	 0.1213 	 0.9760 	 0.0000 	 1.0000 	 0.2253 	 0.9317
[2023/07/09 22:21:37] - 2 	 0.010 	 10.8 	 0.1400 	 0.9740 	 0.0000 	 1.0000 	 0.2349 	 0.9304
[2023/07/09 22:21:48] - 3 	 0.010 	 10.8 	 0.1535 	 0.9720 	 0.0000 	 1.0000 	 0.2513 	 0.9266
[2023/07/09 22:21:59] - 4 	 0.010 	 10.9 	 0.2078 	 0.9640 	 0.0000 	 1.0000 	 0.2770 	 0.9214
[2023/07/09 22:22:10] - 5 	 0.010 	 11.0 	 0.2614 	 0.9500 	 0.0000 	 1.0000 	 0.3144 	 0.9141
[2023/07/09 22:22:21] - 6 	 0.010 	 10.8 	 0.3711 	 0.9220 	 0.0000 	 1.0000 	 0.3847 	 0.8991
[2023/07/09 22:22:31] - 7 	 0.010 	 10.8 	 0.4538 	 0.8700 	 0.0000 	 1.0000 	 0.5276 	 0.8669
[2023/07/09 22:22:42] - 8 	 0.010 	 10.8 	 0.7916 	 0.6700 	 0.0000 	 1.0000 	 0.9439 	 0.7586
[2023/07/09 22:22:53] - 9 	 0.010 	 10.8 	 1.4771 	 0.4540 	 0.0000 	 1.0000 	 2.5574 	 0.4016
[2023/07/09 22:23:04] - 10 	 0.001 	 10.8 	 3.1028 	 0.2920 	 0.0000 	 1.0000 	 2.5620 	 0.3949
[2023/07/09 22:23:15] - 11 	 0.001 	 10.8 	 4.0416 	 0.2360 	 0.0000 	 1.0000 	 2.8507 	 0.3729
[2023/07/09 22:23:25] - 12 	 0.001 	 10.8 	 4.8811 	 0.1980 	 0.0000 	 1.0000 	 3.2337 	 0.3368
[2023/07/09 22:23:25] - ----------- Model Recovering --------------
[2023/07/09 22:23:26] - Epoch 	 lr 	 Time 	 TrainLoss 	 TrainACC 	 PoisonLoss 	 PoisonACC 	 CleanLoss 	 CleanACC
[2023/07/09 22:23:37] - 1 	 0.200 	 11.0 	 1.0719 	 0.1980 	 0.0000 	 1.0000 	 2.0311 	 0.4370
[2023/07/09 22:23:48] - 2 	 0.200 	 11.0 	 0.9892 	 0.1920 	 0.0000 	 1.0000 	 1.3383 	 0.5432
[2023/07/09 22:23:59] - 3 	 0.200 	 11.0 	 0.7809 	 0.2320 	 0.0018 	 1.0000 	 1.1726 	 0.6138
[2023/07/09 22:24:10] - 4 	 0.200 	 11.0 	 0.4892 	 0.2500 	 0.3161 	 0.8770 	 1.0972 	 0.6552
[2023/07/09 22:24:21] - 5 	 0.200 	 11.1 	 0.4130 	 0.2860 	 0.6548 	 0.6051 	 1.0409 	 0.6662
[2023/07/09 22:24:32] - 6 	 0.200 	 11.0 	 0.3691 	 0.3060 	 0.7871 	 0.5084 	 0.9843 	 0.6822
[2023/07/09 22:24:43] - 7 	 0.200 	 11.0 	 0.3262 	 0.3460 	 0.8479 	 0.4791 	 0.9089 	 0.7053
[2023/07/09 22:24:54] - 8 	 0.200 	 11.0 	 0.2963 	 0.3760 	 0.8369 	 0.4904 	 0.8691 	 0.7157
[2023/07/09 22:25:05] - 9 	 0.200 	 11.0 	 0.2777 	 0.3900 	 0.8192 	 0.5090 	 0.8226 	 0.7324
[2023/07/09 22:25:16] - 10 	 0.200 	 11.0 	 0.2485 	 0.4320 	 0.7842 	 0.5340 	 0.7765 	 0.7497
[2023/07/09 22:25:27] - 11 	 0.200 	 11.0 	 0.2337 	 0.4500 	 0.7554 	 0.5562 	 0.7283 	 0.7666
[2023/07/09 22:25:38] - 12 	 0.200 	 11.0 	 0.2140 	 0.4900 	 0.6922 	 0.6044 	 0.7022 	 0.7752
[2023/07/09 22:25:49] - 13 	 0.200 	 11.0 	 0.2043 	 0.5140 	 0.6542 	 0.6317 	 0.6718 	 0.7870
[2023/07/09 22:26:00] - 14 	 0.200 	 11.0 	 0.1807 	 0.5340 	 0.6128 	 0.6598 	 0.6517 	 0.7951
[2023/07/09 22:26:11] - 15 	 0.200 	 11.0 	 0.1724 	 0.5440 	 0.5820 	 0.6873 	 0.6342 	 0.8018
[2023/07/09 22:26:22] - 16 	 0.200 	 11.0 	 0.1729 	 0.5780 	 0.5754 	 0.6968 	 0.6084 	 0.8121
[2023/07/09 22:26:33] - 17 	 0.200 	 11.1 	 0.1532 	 0.6180 	 0.5683 	 0.7027 	 0.5930 	 0.8176
[2023/07/09 22:26:44] - 18 	 0.200 	 11.0 	 0.1476 	 0.6120 	 0.5614 	 0.7083 	 0.5766 	 0.8244
[2023/07/09 22:26:55] - 19 	 0.200 	 11.0 	 0.1510 	 0.6380 	 0.5674 	 0.7069 	 0.5601 	 0.8312
[2023/07/09 22:27:06] - 20 	 0.200 	 11.1 	 0.1439 	 0.6520 	 0.5788 	 0.6988 	 0.5417 	 0.8402
[2023/07/09 22:27:07] - ----------- Backdoored Model Pruning --------------
[2023/07/09 22:27:07] - Pruned Number 	 Layer Name 	 Neuron Idx 	 Mask 	 PoisonLoss 	 PoisonACC 	 CleanLoss 	 CleanACC
[2023/07/09 22:27:17] - 0 	 None     	 None     	 0.0001 	 1.0000 	 0.2157 	 0.9340
[2023/07/09 22:27:27] - 12.00 	 layer4.1.bn2 	 188 	 0.0 	 0.0122 	 0.9986 	 0.2104 	 0.9347
[2023/07/09 22:27:37] - 14.00 	 layer4.1.bn2 	 12 	 0.05 	 0.0139 	 0.9982 	 0.2092 	 0.9340
[2023/07/09 22:27:46] - 18.00 	 layer3.0.bn2 	 230 	 0.1 	 0.0601 	 0.9876 	 0.2065 	 0.9338
[2023/07/09 22:27:56] - 21.00 	 bn1 	 5 	 0.15000000000000002 	 0.9935 	 0.4796 	 0.2074 	 0.9340
[2023/07/09 22:28:06] - 24.00 	 layer4.0.bn2 	 82 	 0.2 	 2.8242 	 0.0661 	 0.2297 	 0.9280
[2023/07/09 22:28:16] - 28.00 	 layer3.0.bn1 	 106 	 0.25 	 3.2791 	 0.0424 	 0.2292 	 0.9270
[2023/07/09 22:28:26] - 32.00 	 layer4.1.bn2 	 152 	 0.30000000000000004 	 4.2908 	 0.0172 	 0.2295 	 0.9277
```

## Backdoor Model Weights
You can directly download the pre-trained backdoored model weights with the links below:  

| Attacks | Paper Name | Baidu Weight Source (pwd: 1212) | Google Weight Source  |
|:---:|:---:|:---:|:---:|
| Badnets | Badnets: Evaluating Backdooring Attacks on Deep Neural Networks | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| Trojan | Trojaning attack on Neural Networks | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| Blend | Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| CL | Label-Consistent Backdoor Attacks | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| SIG | A New Backdoor Attack in Cnns by Training Set Corruption without Label Poisoning | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| Dynamic | Input-Aware Dynamic Backdoor Attack | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| WaNet | WaNet - Imperceptible Warping-based Backdoor Attack | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| FC | Poison Frog! Targeted Clean-label Backdoor Attacks on Neural Networks | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| DFST | Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxifcation | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| AWP | Can Adversarial Weight Perturbations Inject Neural Backdoors | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| LIRA | LIRA: Learnable, Imperceptible and Robust Backdoor Attacks | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |
| A-Blend | Circumventing Backdoor Defense that are Based on Latent Separability  | [Baidu Drive](https://pan.baidu.com/s/1LXZuvb06als1D025eK04_Q) | [Google Drive]() |


## Citation
If you use this code in your work, please cite the accompanying paper:

```
@inproceedings{
li2023reconstructive,
title={Reconstructive Neuron Pruning for Backdoor Defense},
author={Yige Li and Xixiang Lyu and Xingjun Ma and Nodens Koren and Lingjuan Lyu and Bo Li and Yu-Gang Jiang},
booktitle={ICML},
year={2023},
}
```

## Acknowledgements
As this code is reproduced based on the open-sourced code [ANP](https://github.com/csdongxian/ANP_backdoor) and [DCB](https://github.com/HanxunH/CognitiveDistillation), the authors would like to thank their contribution and help. 



## Backdoor-related repo:
  - Dynamic Attack: https://github.com/VinAIResearch/input-aware-backdoor-attack-release
  - STRIP: https://github.com/garrisongys/STRIP
  - NAD: https://github.com/bboylyg/NAD
  - ABL: https://github.com/bboylyg/ABL
  - Frequency: https://github.com/YiZeng623/frequency-backdoor
  - NC: https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses/neural_cleanse
  - BackdoorBox: https://github.com/THUYimingLi/BackdoorBox
  - BackdoorBench:https://github.com/SCLBD/BackdoorBench
