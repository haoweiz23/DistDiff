# Distribution-Aware Data Expansion with Diffusion Models
This is the official repository of  [Distribution-Aware Data Expansion with Diffusion Models (DistDiff)](https://arxiv.org/abs/2403.06741).

<hr />

## 1. Highlights

![main figure](figure/intro.png)


> **<p align="justify"> Abstract:** *The scale and quality of a dataset significantly impact the performance of deep models. However, acquiring large-scale annotated datasets is both a costly and time-consuming endeavor. To address this challenge, dataset expansion technologies aim to automatically augment datasets, unlocking the full potential of deep models. Current data expansion techniques include image transformation and image synthesis methods. Transformation-based methods introduce only local variations, leading to limited diversity. In contrast, synthesis-based methods generate entirely new content, greatly enhancing informativeness. However, existing synthesis methods carry the risk of distribution deviations, potentially degrading model performance with out-of-distribution samples. In this paper, we propose DistDiff, a training-free data expansion framework based on the distribution-aware diffusion model. DistDiff constructs hierarchical prototypes to approximate the real data distribution, optimizing latent data points within diffusion models with hierarchical energy guidance. We demonstrate its capability to generate distribution-consistent samples, significantly improving data expansion tasks. DistDiff consistently enhances accuracy across a diverse range of datasets compared to models trained solely on original data. Furthermore, our approach consistently outperforms existing synthesis-based techniques and demonstrates compatibility with widely adopted transformation-based augmentation methods. Additionally, the expanded dataset exhibits robustness across various architectural frameworks.* </p>

<hr />

## 2. Main Contributions
1) We introduce a novel diffusion-based data expansion algorithm, named **DistDiff**, which facilitates distribution consistent data augmentation without requiring retraining.
2) By leveraging hierarchical prototypes to approximate data distribution, we propose an effective distribution-aware energy guidance at both class and group levels in the diffusion sampling process.
3) The experimental results illustrate that our DistDiff is capable of generating high-quality samples, surpassing existing image transformation and synthesis methods significantly.

<hr />

## 3. Method Pipeline

<p align="center">
  <img src="figure/overview.jpg"  style="height:400px; max-width:100%;" />
</p>

Please refer to the [paper](https://arxiv.org/abs/2403.06741) for more technical details.

<hr />

## 4. How to Run Experiments?
### 4.1  Setup & Preparation
Please follow the instructions detailed in [INSTALL.md](INSTALL.md) to install necessary libraries and prepare all datasets.

<hr />

### 4.2 Run Experiments
**1) obtain guide model**
* First, you can obtain the guide model by training the model on the target dataset, or you can directly use the pretrained model by setting the `pretrained` parameter to True in the `create_model` function.
```
# Taking caltech-101 as an example, these commands will train a resnet50 model from scratch.

python train_original.py caltech-101 resnet50 0.1 False 0
```
* The trained weights will be saved in `checkpoint/caltech-101/resnet50_unpretrained_lr0.1`.
  
**2) Data Expansion**
* We utilize 4 distinct GPUs to execute the entire program. This is achieved by modifying CUDA_VISIBLE_DEVICES=N and setting --split N for each GPU, where N ranges from 0 to 3.
```
# Taking caltech-101 as an example, these command will generate 5x expanded datasets.

sh scripts/exps/expand_diff.sh 5 0 0 &
sh scripts/exps/expand_diff.sh 5 1 1 &
sh scripts/exps/expand_diff.sh 5 2 2 &
sh scripts/exps/expand_diff.sh 5 3 3 
```
* Please note that if your system doesn't possess 4 GPUs, you can modify the `total_split` parameter in `scripts/exps/expand_diff.sh` to reflect the actual number of GPUs you have, and make corresponding adjustments to the `SPLIT` and `GPU`.

**3) Training and Evaluation**
* After data expansion, use the following script to train and evaluate the backbone on the expanded datasets.

```
sh scripts/exps/train_expand.sh ${DATASET} ${BACKBONE} 0.1 ${EXP} False ${GPU}
```

* You can also execute previous steps jointly by:
```
# this scripts will generate 5 time expanded caltech-101 by our DistDiff and train resnet-50 model on the expanded data.

sh single_exp.sh
```

* Parse the evaluation accuracy

```
python parse_logs.py /path/to/exp --multi
```

<hr />

## Citation

If you find our work inspiring in your research, please cite our work.

```
@article{zhu2024distribution,
  title={Distribution-Aware Data Expansion with Diffusion Models},
  author={Zhu, Haowei and Yang, Ling and Yong, Jun-Hai and Yin, Hongzhi and Jiang, Jiawei and Xiao, Meng and Zhang, Wentao and Wang, Bin},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={102768--102795},
  year={2024}
}
```

<hr />


