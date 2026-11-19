# 6.7 Foundation models

Part of [6. Self-supervised and transfer learning](README.md). [Slides 51-62](slides.pdf#page=51).

> [!TIP]
> **Key takeaways**
>
> - Foundation models combine the ideas of this topic (self-distillation, masked modeling, image-text alignment) and scale them up.
> - At this scale, curated data and model size matter as much as the loss.
> - In-domain foundation models exist for medical imaging; start from them as you would from ImageNet weights.

## Notes

A *foundation model* is a general feature extractor trained on vast amounts of data, meant to be adapted to many downstream tasks. The methods of the previous pages are their building blocks; the recent models combine them and scale them up.

### iBOT

iBOT (Zhou et al., 2022) combines the contrastive (self-distillation) loss of DINO with a generative, masked-modeling loss as in MAE. The difference with MAE is subtle but important: the masked patches are predicted **in feature space**, not in pixel space. The student sees a masked image and must predict, for every masked patch, the teacher's features of that patch; the teacher acts as an *online tokenizer*. Because there is no pixel decoder, training is cheaper, which allows larger batches. The mask tokens move from the decoder (MAE) to the encoder.

### DINOv2 and DINOv3

DINOv2 (Oquab et al., 2024) and DINOv3 (Siméoni et al., 2025) show what these losses do **at scale**:

- the losses are essentially those of iBOT (DINO plus masked feature prediction), with extra regularization terms;
- the models are large: the biggest DINOv3 model is a ViT with 7 billion parameters; for comparison, ViT-L has 300 million;
- the training data is **curated**: once you can collect billions of images, quality matters more than quantity, so the data is filtered and balanced towards a diverse set of concepts.

The result are frozen features that are strong out of the box for classification, segmentation, depth estimation and retrieval, often without any fine-tuning. Many recent models in medical imaging start from DINOv2 or DINOv3 weights.

### Segment Anything (SAM)

Segment Anything (Kirillov et al., 2023) is a foundation model for *segmentation*. It doesn't know classes, only objects: given a prompt (a point, a box or a rough mask), it returns the mask of the object at that location. It was trained on more than **one billion masks** in 11 million images, collected with a *data engine*: the model helped annotators label images, and the improved model was then used to label more, until masks were generated fully automatically.

SAM 2 (Ravi et al., 2024) extends this to **video**, using a memory of earlier frames to track the segmented object over time.

### Foundation models in medical imaging

Medical images are a domain where labels are scarce and costly, so domain-specific self-supervised pretraining pays off. Examples from our research group:

- **GastroNet**: ResNet-50 models pretrained with MoCo, SimCLR and DINO on over 5 million endoscopy images from Dutch hospitals; DINO mainly benefits vision transformers. In-domain pretraining outperforms ImageNet pretraining on tasks such as Barrett's neoplasia and colon polyp detection.
- **SurgeNet**: a MetaFormer pretrained with DINO on 4.7 million frames of surgical video, which improves anatomy recognition, surgical phase recognition and critical-view-of-safety classification.
- **SPECTRE**: a transformer for 3D CT scans, pretrained with DINO-style self-distillation and SigLIP-style alignment with radiology reports, on openly available CT data. It is evaluated on tasks such as lung nodule classification, liver metastasis detection and abdominal organ segmentation.

## In practice

Where to find the models of this topic:

| Model | Where | Note |
| --- | --- | --- |
| DINOv2 | `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")` | Sizes ViT-S to ViT-g |
| DINOv3 | [Hugging Face](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | Accept the license on the model page first |
| CLIP and SigLIP | [OpenCLIP](https://github.com/mlfoundations/open_clip), Hugging Face | For zero-shot use and image-text tasks |
| SAM 2 | [GitHub](https://github.com/facebookresearch/sam2) | Prompted image and video segmentation |
| SurgeNet | [Hugging Face](https://huggingface.co/TimJaspersTue/SurgeNetModels) | Surgical video |
| GastroNet | [Hugging Face](https://huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights) | Endoscopy |
| SPECTRE | [GitHub](https://github.com/cclaess/SPECTRE) | 3D CT |

Ideas for the [final assignment](../../final-assignment/README.md), which allows any public weights:

- **Dice and Real-time tracks:** replace the ImageNet encoder of the baseline by a DINOv2/v3 ViT or by SurgeNet, which was pretrained on surgical video like ATLAS. For the Real-time track, distill such a model into a small one.
- **Consistency track:** SAM 2's memory of earlier frames is an example of the causal temporal processing this track asks for.
- Whatever you use, the weights must be inside your submission image: the server has no internet.

## Check your understanding

<details class="question" markdown="1">
<summary>Both MAE and iBOT predict masked patches. What is the difference?</summary>

MAE reconstructs the **pixels** of masked patches with a decoder. iBOT predicts the teacher's **features** of the masked patches, so it learns more abstract targets, needs no pixel decoder, and is cheaper per image. In iBOT the mask tokens enter the encoder; in MAE only the decoder sees them.

</details>

<details class="question" markdown="1">
<summary>Why does DINOv2 curate its training data instead of using all images it can collect?</summary>

Uncurated web data is dominated by a few common concepts and contains many near-duplicates. The model then spends most of its capacity on those, and rare concepts are underrepresented. Filtering and balancing gives a more diverse set of concepts for the same compute, and better features.

</details>

<details class="question" markdown="1">
<summary>Can SAM replace a semantic segmentation model for the Dice track?</summary>

Not directly. SAM returns masks of objects at a prompt, but no class labels, and it needs prompts. You would still need something that finds where to prompt and which class each mask is. Its features, or the masks as an extra input, can help a semantic segmentation model, though.

</details>

## Further reading

- J. Zhou et al., "iBOT: Image BERT pre-training with online tokenizer," *ICLR*, 2022. [arXiv](https://arxiv.org/abs/2111.07832)
- M. Oquab et al., "DINOv2: Learning robust visual features without supervision," *TMLR*, 2024. [arXiv](https://arxiv.org/abs/2304.07193)
- O. Siméoni et al., "DINOv3," 2025. [arXiv](https://arxiv.org/abs/2508.10104)
- T. Darcet, M. Oquab, J. Mairal and P. Bojanowski, "Vision transformers need registers," *ICLR*, 2024. [arXiv](https://arxiv.org/abs/2309.16588). Explains artifacts in the attention maps of large ViTs, and a simple fix.
- A. Kirillov et al., "Segment anything," *ICCV*, 2023. [arXiv](https://arxiv.org/abs/2304.02643)
- N. Ravi et al., "SAM 2: Segment anything in images and videos," *ICLR*, 2025. [arXiv](https://arxiv.org/abs/2408.00714)
- T. G. W. Boers et al., "Foundation models in gastrointestinal endoscopic AI: Impact of architecture, pre-training approach and data efficiency," *Medical Image Analysis*, 2024. [Paper](https://pure.amsterdamumc.nl/en/publications/foundation-models-in-gastrointestinal-endoscopic-ai-impact-of-arc/)
- T. J. M. Jaspers et al., "Scaling up self-supervised learning for improved surgical foundation models," *Medical Image Analysis*, 2025. [arXiv](https://arxiv.org/abs/2501.09436)
- C. Claessens, C. Viviers, G. D'Amicantonio, E. Bondarev and F. van der Sommen, "Scaling self-supervised and cross-modal pretraining for volumetric CT transformers," *CVPR*, 2026. [arXiv](https://arxiv.org/abs/2511.17209)

Next: [7. Probabilistic and generative modeling](../07-generative-models/README.md).
