# 6.3 Evaluating representations

Part of [6. Self-supervised and transfer learning](README.md). [Slides 26-30](slides.pdf#page=26).

> [!TIP]
> **Key takeaways**
>
> - We evaluate the representation a model learned, not its pretext task.
> - Qualitative checks (retrieval, t-SNE) build intuition; k-NN, linear probing and fine-tuning give comparable numbers.
> - Frozen-feature tests and fine-tuning measure different things, so report more than one.

## Notes

A pretext task is only a means to an end. To know whether pretraining taught the model something useful, we evaluate the *representations* it produces: the feature vectors (embeddings) of its layers.

### Qualitative evaluation

Qualitative methods don't give a number, but let you *look* at what the model learned.

- **Compare representations in different layers.** Visualize the features of early and late layers and check whether they respond to edges and textures, or to object parts and whole objects.
- **Image retrieval.** Compute the feature vector of every image in a dataset. For a query image, compute the Euclidean (or cosine) distance to all other feature vectors and rank the images by distance. If the nearest neighbors of a dog are other dogs, the representation captures semantics.
- **Dimensionality reduction.** Project the high-dimensional feature vectors to two dimensions with t-SNE or PCA, and color the points by class. Clusters per class indicate a semantically organized representation.

> [!WARNING]
> A 2D projection squeezes hundreds of dimensions into two. t-SNE in particular distorts distances: cluster sizes and the distances between clusters mean little. Use these plots for intuition, not as evidence.

### Quantitative evaluation

Quantitative methods give objective, reproducible numbers, so you can compare models and methods.

- **k-NN classification.** Classify each test image by a majority vote of its k nearest training images in feature space. No training is needed at all.
- **Linear probing.** Freeze the encoder and train only a linear classifier on its features. A good linear-probe accuracy means the information is not only present in the features, but also linearly accessible.
- **Fine-tuning.** Use the pretrained encoder as initialization for a downstream task and fine-tune it, as in [6.1](1-transfer-learning.md). This measures how good a starting point the model is; it is closest to how foundation models are used in practice, for example a CT foundation model fine-tuned for organ segmentation, cancer detection or disease prognosis.

k-NN and linear probing test the frozen representation; fine-tuning tests the pretrained weights as a whole. A method can do well on one and worse on another, so papers usually report several.

## Check your understanding

<details class="question" markdown="1">
<summary>Two pretrained models have the same fine-tuning accuracy, but model A has a much higher linear-probe accuracy. What does that tell you?</summary>

Model A's frozen features already make the classes linearly separable; model B contains the information too, but only becomes useful after its weights are adapted. Model A is the better choice when you can't fine-tune, for example with very little data or when one frozen backbone has to serve many tasks.

</details>

<details class="question" markdown="1">
<summary>Why is k-NN evaluation attractive for comparing self-supervised methods?</summary>

It has no training and almost no hyperparameters (only k and the distance), so differences in the result come from the representation, not from how well the classifier on top was tuned.

</details>

## Further reading

- G. Alain and Y. Bengio, "Understanding intermediate layers using linear classifier probes," *ICLR workshop*, 2017. [arXiv](https://arxiv.org/abs/1610.01644)
- P. Goyal, D. Mahajan, A. Gupta and I. Misra, "Scaling and benchmarking self-supervised visual representation learning," *ICCV*, 2019. [arXiv](https://arxiv.org/abs/1905.01235). Proposes a standard benchmark and shows how much evaluation choices matter.
- L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *JMLR*, 2008. [Paper](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
- M. Wattenberg, F. Viégas and I. Johnson, "How to use t-SNE effectively," *Distill*, 2016. [Article](https://distill.pub/2016/misread-tsne/). Interactive examples of what t-SNE plots do and don't show.

Next: [6.4 Masked image modeling](4-masked-image-modeling.md).
