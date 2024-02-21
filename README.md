# p-Laplacian Adaptation for Generative Pre-trained Vision-Language Models (AAAI'24 Oral)
<img src="./img/p-adapter.png" width="700">

## Introduction
We present a novel modeling framework that recasts adapter tuning after attention as a graph message passing process on attention graphs, where the projected query and value features and attention matrix constitute the node features and the graph adjacency matrix, respectively. Within this framework, tuning adapters in VLMs necessitates handling heterophilic graphs, owing to the disparity between the projected query and value space.

To address this challenge, we propose a new adapter architecture, $p$-adapter, which employs [$p$-Laplacian message passing](https://arxiv.org/abs/2111.07337) in GNNs. Specifically, the attention weights are re-normalized based on the features, and the features are then aggregated using the calibrated attention matrix, enabling the dynamic exploitation of information with varying frequencies in the heterophilic attention graphs.

This is the official Pytorch implementation of [p-Adapter](https://arxiv.org/abs/2312.10613).

## Installation
```bash
# Download pretrained models BLIP and bert_base_uncased in "./download_model/"

# Install python dependencies
pip install -r requirements.txt
```

## VL Tasks

### COCO Caption
```bash
# Download COCO datasets from the original websites in "./dataset/".

bash train_coco_caption.sh
```

### SNLI_VE
```bash
# Download SNLI_VE dataset from the original websites in "./dataset/".

bash train_snli_ve.sh
```

### VQA
```bash
# Download VQA v2 dataset from the original websites in "./dataset/".

bash train_vqa.sh
```

## Acknowledgement
This repo is adapted from [BLIP](https://github.com/salesforce/BLIP).

## Citation
```bibtex
@article{wu2023p,
  title={p-Laplacian Adaptation for Generative Pre-trained Vision-Language Models},
  author={Wu, Haoyuan and Zhang, Xinyun and Xu, Peng and Liao, Peiyu and Yao, Xufeng and Yu, Bei},
  journal={arXiv preprint arXiv:2312.10613},
  year={2023}
}
```
