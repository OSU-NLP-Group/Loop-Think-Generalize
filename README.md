# [COLM '26] Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers (https://arxiv.org/abs/2604.07822)

>We study implicit reasoning, i.e. the ability to combine knowledge or rules within a single forward pass. While transformer-based large language models store substantial factual knowledge and rules, they often fail to compose this knowledge for implicit multi-hop reasoning, suggesting a lack of compositional generalization over their parametric knowledge. To address this limitation, we study recurrent-depth transformers, which enables iterative computation over the same transformer layers. We investigate two compositional generalization challenges under the implicit reasoning scenario: systematic generalization, i.e. combining knowledge that is never used for compositions during training, and recursion extrapolation, i.e. generalizing from limited reasoning depth (e.g. training on up to 5-hop) to deeper compositions (e.g. 10-hop). Through controlled studies with models trained from scratch, we show that while vanilla transformers struggle with both generalization challenges, recurrent-depth transformers can effectively make such generalization. For systematic generalization, we find that this ability emerges through a three-stage grokking process, transitioning from memorization to in-distribution generalization and finally to systematic generalization, supported by mechanistic analysis. For recursion extrapolation, we show that generalization beyond training depth can be unlocked by scaling inference-time recurrence, with more iterations enabling deeper reasoning. We further study how training strategies affect extrapolation, providing guidance on training recurrent-depth transformers, and identify a key limitation, overthinking, where excessive recurrence degrades predictions and limits generalization to very deep compositions.

## Downloading Artifacts

- Data and select checkpoints from our experiments can be downloaded [here](https://drive.google.com/drive/folders/1AQvp8erdtRRn9_ix7E1ssmyWarwFNLVU?usp=sharing)
- The depth exploration experiments, the final models (used in Figure 5 of the paper) are provided in checkpoints/multi_hop/final_models whereas the 12-hop generalzed models (used in Figure 6 of the paper) can be downloaded from in checkpoints/multi_hop/12_hop

### Checkpoint Structure for Systematicity

For the systematicity experiments, **`r<X>_l<Y>`** Indicates a model trained with `<X>` recurrence steps and `<Y>` transformer layers in the recurrent block.

Within each systematicity directory, checkpoint files are named `checkpoint_epoch_<x>.pt`, where `x` is the training epoch. The number of checkpoints per directory depends on the model architecture:
* **Recurrent Models (`r != 1`):** Contain 3 checkpoints, corresponding to Train, In-Distribution (ID), and Out-Of-Distribution (OOD) generalization.
* **Vanilla Transformers (`r = 1`):** Contain 2 checkpoints (Train and ID). As noted in our paper, vanilla transformers do not exhibit OOD generalization.

## Model Training

### Systematicity

- For the systematicity experiments, set the necessary parameters such as --recurrence and --num_recurrent_layers in the driver file train_systematicity.py 
- Most other parameters are set to defaults used to reproduce results in the paper including --seed (42)

`python train_systematicity.py`

### Depth Extrapolation

- Similar to systematicity, set the necessary parameters in train_extrpolation.py 
- Vary the --recurrence_type to denote fixed or dynamic recurrence 
- Most other parameters are set to defaults used to reproduce results in the paper including --seed (42)

`python train_extrapolation.py`

## Evaluation

- For experiment with varying recurrent-depths at inference time, set the required parameters such as --checkpoint_dir, --model_name, and --recurrence_range and run:

`python inference_extrapolation.py`

- Similarly, for experiments with adaptive halting (r*) , run:

`python inference_extrapolation_adaptive.py`

## Optional Data Generation

### Depth Extrapolation

Set the required parameters such as NUM_ENTITY_IN, NUM_RELATION, max_hop, and a temporary data_dir and then run - 

`python getNhopfact.py`

Next, set the temporary data_dir from the previous run as the base_path, and a new output_dir and then run - 

`python create_dataset.py`

This should create a new dataset with the defined characteristics (entities, relations etc.) in the output_dir

### Systematicity

For systematicity experiments using 2-hop composition, we use the data generation pipeline from the [GrokkedTransformers](https://github.com/OSU-NLP-Group/GrokkedTransformer) repository. In all our experiments, the inferred/atomic ratio is 7.2 with 2000 entities and 200 relations. Please look in the GrokkedTransformers repo for more details on how to re-generate data.

## Logit Margin Analysis

To reproduce Figure 8 from the paper set the required parameters in model_specs and other necessary paths and run - 

`python logit_margin_analysis.py`

## Citation
```
@article{kohli2026loop,
  title={Loop, Think, \& Generalize: Implicit Reasoning in Recurrent-Depth Transformers},
  author={Kohli, Harsh and Parthasarathy, Srinivasan and Sun, Huan and Yao, Yuekun},
  journal={arXiv preprint arXiv:2604.07822},
  year={2026},
  doi={10.48550/arXiv.2604.07822},
  url={https://arxiv.org/abs/2604.07822}
}
```