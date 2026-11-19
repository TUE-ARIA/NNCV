# 6. Self-supervised and transfer learning

Labels are the bottleneck of supervised learning. Annotating images is slow, often needs experts or special equipment, does not scale to millions of images, and is sometimes not even well defined: two experts can draw different outlines of the same tumor. Meanwhile, huge amounts of unlabeled images are available.

This topic answers two questions:

- Can we **reuse a model** that was trained for another task, instead of starting from scratch? (*transfer learning*)
- Can we **learn from unlabeled data**, by letting the data provide its own supervision? (*self-supervised learning*)

Together they explain how today's *foundation models* are built and why almost every computer-vision project, including the final assignment, starts from one.

## Learning goals

After this topic, you can:

- choose between using a pretrained model as a frozen feature extractor and fine-tuning it, based on the size of your dataset and how similar it is to the pretraining data;
- explain how predictive pretext tasks (context prediction, jigsaw puzzles, colorization, rotation) create labels from the data itself, and recognize the shortcuts a model can take to solve them;
- evaluate a learned representation qualitatively (retrieval, t-SNE) and quantitatively (k-NN, linear probing, fine-tuning);
- explain masked image modeling, and why masked autoencoders suit vision transformers;
- derive the InfoNCE loss and explain how MoCo, SimCLR, BYOL and DINO learn invariant representations, and how methods without negative pairs avoid collapse;
- explain how image-text models such as CLIP and SigLIP learn from weak supervision, and how that gives zero-shot classification;
- describe how foundation models such as DINOv2/v3 and Segment Anything combine these ideas at scale.

## Subtopics

Read them in order: each one builds on the previous.

| # | Subtopic | Slides |
| --- | --- | --- |
| 6.1 | [Transfer learning](1-transfer-learning.md) | 2-10 |
| 6.2 | [Predictive self-supervised learning](2-predictive-ssl.md) | 11-25 |
| 6.3 | [Evaluating representations](3-evaluation.md) | 26-30 |
| 6.4 | [Masked image modeling](4-masked-image-modeling.md) | 31-34 |
| 6.5 | [Contrastive learning and self-distillation](5-contrastive-learning.md) | 35-47 |
| 6.6 | [Weak supervision and vision-language models](6-weak-supervision.md) | 48-50, 54 |
| 6.7 | [Foundation models](7-foundation-models.md) | 51-62 |

## Lectures and slides

This topic takes one lecture; the [schedule](../../README.md#schedule) shows when.

| Lecture | Covers | Slides |
| --- | --- | --- |
| 1 | 6.1-6.7 | [PDF](slides.pdf) |

- **Notebook:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TUE-ARIA/NNCV/blob/main/topics/06-self-supervised-learning/06-self-supervised-learning.ipynb) or [download it](06-self-supervised-learning.ipynb). It is still a placeholder.

## Summary

- **Transfer learning:** reuse a pretrained network for a new task, as a feature extractor or by fine-tuning it. Start here unless you have a very good reason not to.
- **Self-supervised learning:** create your own labels from the data, for example by removing information and learning to restore it. Modern methods learn representations with generative (masked modeling) or contrastive (self-distillation) objectives, and the strongest models combine both.
- **Weakly supervised learning:** use cheap, imperfect labels such as image captions. Learning from text gives zero-shot recognition.

## Further reading

- R. Balestriero et al., "A cookbook of self-supervised learning," 2023. [arXiv](https://arxiv.org/abs/2304.12210). A practical overview of the whole field, from pretext tasks to DINO, with training tips.
- L. Jing and Y. Tian, "Self-supervised visual feature learning with deep neural networks: A survey," *IEEE TPAMI*, 2020. [arXiv](https://arxiv.org/abs/1902.06162). A survey of the earlier, pretext-task-based methods.
- R. Bommasani et al., "On the opportunities and risks of foundation models," 2021. [arXiv](https://arxiv.org/abs/2108.07258). Where the term *foundation model* comes from.

Each subtopic lists further reading on its own subject.

Next: [6.1 Transfer learning](1-transfer-learning.md).
