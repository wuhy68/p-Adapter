# p-Laplacian Adaptation for Generative Pre-trained Vision-Language Models (AAAI'24 Oral)

## Introduction
We present a novel modeling framework that recasts adapter tuning after attention as a graph message passing process on attention graphs, where the projected query and value features and attention matrix constitute the node features and the graph adjacency matrix, respectively. Within this framework, tuning adapters in VLMs necessitates handling heterophilic graphs, owing to the disparity between the projected query and value space.

To address this challenge, we propose a new adapter architecture, $p$-adapter, which employs [$p$-Laplacian message passing](https://arxiv.org/abs/2111.07337) in GNNs. Specifically, the attention weights are re-normalized based on the features, and the features are then aggregated using the calibrated attention matrix, enabling the dynamic exploitation of information with varying frequencies in the heterophilic attention graphs.

<img src="./img/p-adapter.pngf" width="700">

This is the official Pytorch implementation of [p-Adapter](https://arxiv.org/abs/2312.10613). 



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
