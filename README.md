# Data augmentation with norm-AE and selective Pseudo-Labelling for Unsupervised Domain Adaptation

## Abstract
We address the Unsupervised Domain Adaptation (UDA) problem in image classification from a new perspective. In contrast to most existing works which either align the data distributions or learn domain-invariant features, we directly learn a unified classifier for both the source and target domains in the high-dimensional homogeneous feature space without explicit domain alignment. To this end, we employ the effective Selective Pseudo-Labelling (SPL) technique to take advantage of the unlabelled samples in the target domain. Surprisingly, data distribution discrepancy across the source and target domains can be well handled by a computationally simple classifier (e.g., a shallow Multi-Layer Perceptron) trained in the original feature space. Besides, we propose a novel generative model \textit{norm-AE} to generate synthetic features for the target domain as a data augmentation strategy to enhance the classifier training. Experimental results on several benchmark datasets demonstrate the pseudo-labelling strategy itself can lead to comparable performance to many state-of-the-art methods whilst the use of \textit{norm-AE} for feature augmentation can further improve the performance in most cases. As a result, our proposed methods (i.e. \textit{naive-SPL} and \textit{norm-AE-SPL}) can achieve comparable performance with state-of-the-art methods with the average accuracy of 93.4\% and 90.4\% on Office-Caltech and ImageCLEF-DA datasets, and achieve competitive performance on Digits, Office31 and Office-Home datasets with the average accuracy of 97.2\%, 87.6\% and 68.6\% respectively.

## Citation
@article{wang23augmentation,\
 author = {Wang, Q. and Meng, F. and Breckon, T.P.},\
 title = {Data Augmentation with norm-AE and Selective Pseudo-Labelling for Unsupervised Domain Adaptation},\
 journal = {Neural Networks},\
 volume={161},
 pages={614--625},
 year = {2023},\
 month = {February},\
 publisher = {Elsevier},\
 keywords = {unsupervised domain adaptation, data augmentation, variational autoencoder, selective pseudo-labelling},\
 url = {https://breckon.org/toby/publications/papers/wang23augmentation.pdf}, \
 arxiv = {https://arxiv.org/abs/2012.00848}, \
 note = {to appear},\
 category = {imageclass},\
}
## Contact
qian.wang173@hotmail.com
