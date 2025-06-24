# BioVFM-21M: Benchmarking and Scaling Self-Supervised Vision Foundation Models for Biomedical Image Analysis

### Dataset

We curated BioVFM-21M, a large biomedical image dataset consisting of 21 million biomedical images from 61 publicly available data sources. BioVFM-21M is a curated aggregation of publicly available biomedical image datasets. We sincerely thank all original dataset authors and contributors for their valuable work. Please note that we do not host or redistribute any raw data. All images included in BioVFM-21M were obtained from official data sources under their respective licenses. If you are interested in the original datasets, please refer to and download them directly from the official links listed in [`data_sources.pdf`](./dataset/data_sources.pdf). This dataset is provided for research purposes only and should be used in accordance with the terms and conditions of the original data providers. We also provide a complete data list of BioVFM-21M at [google drive](https://drive.google.com/drive/folders/1iJdnG4lkpBwH4JHYdIawWI-DCcgKPcP8?usp=sharing), which contains the filenames of all selected images, and the corresponding raw file paths.

### Pretraining

We follow the default pretraining settings from [DINOv2](https://github.com/facebookresearch/dinov2) and [MAE](https://github.com/facebookresearch/mae), including data augmentation, optimizer, and learning rate schedules. For further implementation details related to model pretraining, please refer to the original repositories. Our pretrained model can be found at [here](https://drive.google.com/drive/folders/1iJdnG4lkpBwH4JHYdIawWI-DCcgKPcP8?usp=sharing).

### Evaluation

We assess model performance using the Area Under the Curve (AUC) score across 12 diagnostic benchmarks from [MedMNIST](https://medmnist.com/). The evaluation can be performed with [`evaluation/scripts/linear_prob_medmnist.sh`](evaluation/scripts/linear_prob_medmnist.sh)

### Scalability analysis

The details of scalability analysis can be found at [`analysis/analyze_results.ipynb`](analysis/analyze_results.ipynb).

### Citation

```
@article{liu2025biovfm,
  title={BioVFM-21M: Benchmarking and Scaling Self-Supervised Vision Foundation Models for Biomedical Image Analysis},
  author={Liu, Jiarun and Zhou, Hong-Yu and Huang, Weijian and Yang, Hao and Song, Dongning and Tan, Tao and Liang, Yong and Wang, Shanshan},
  journal={arXiv preprint arXiv:2505.09329},
  year={2025}
}
```

