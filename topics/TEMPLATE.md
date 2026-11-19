# Writing a topic page

For staff. This file is not part of the website. Each topic folder holds three files:

```text
topics/04-transformers/
├── README.md               the notes: the page on the website
├── 04-transformers.ipynb   the exercises, opened in Colab from the page
└── slides.pdf              the slides, linked from the page
```

When the slides are ready, add `slides.pdf` to the folder and replace "coming soon" on the page with `[PDF](slides.pdf)`. Keep PDFs small (export with compressed images): every version stays in the git history.

To add, merge or reorder topics, change the topic folders, the list in [README.md](README.md), the `nav` in `mkdocs.yml` and the schedule in the [course README](../README.md#schedule).

## Page structure

Copy the skeleton below into the topic's `README.md` and keep the headings, so all topics look the same.

````markdown
# 4. Transformers

- **Slides:** [PDF](slides.pdf)
- **Notebook:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB-LINK) or [download it](04-transformers.ipynb)

## Learning goals

After this topic, you can:

- explain how self-attention mixes information between image patches,
- ...

## Notes

### From convolutions to attention

Text that follows the slides, one `###` section per block of slides.

## Further reading

- A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," *ICLR*, 2021. [arXiv](https://arxiv.org/abs/2010.11929)

Next: [5. Model performance evaluation](../05-evaluation/README.md).
````

The Colab link is `https://colab.research.google.com/github/TUE-ARIA/NNCV/blob/main/topics/<folder>/<folder>.ipynb`.

## Topics with subtopics

A large topic gets one page per **subtopic**, next to an overview page. The notes follow the subjects, not the lectures: a lecture can cover several subtopics, and a subtopic can span two lectures. Topics 3, 4 and 7 show the pattern:

```text
topics/07-generative-models/
├── README.md                       overview: learning goals, subtopics, lectures with their slides
├── 1-probabilistic-modeling.md     notes of subtopic 7.1
├── 2-vaes.md                       notes of subtopic 7.2
├── …
├── slides-1.pdf  slides-2.pdf  slides-3.pdf    one per lecture, linked from the overview
└── 07-vaes.ipynb  07-gans.ipynb  …              optional, one per subtopic or shared
```

- The overview page has the learning goals, a table of subtopics with their notebooks, and a table of lectures with what they cover and their slides.
- A subtopic page has the notes, its notebook (if any) and further reading. Not every subtopic needs a notebook (7.1 has none), and subtopics can share one (topic 4).
- In `mkdocs.yml`, the topic becomes a section: the overview first, then the subtopics. In the sidebar it expands into its subtopics.
- In the course schedule, every lecture of the topic links to the overview page.

## What you can use

- **Maths:** inline `$\mathbf{x}$` and display `$$ ... $$`, written in LaTeX.
- **Figures:** put images in the topic folder and include them with `![Short description](figure.png)`. Add a caption in italics on the line below.
- **Citations in the text:** footnotes, `... as in ViT[^vit].`, with `[^vit]: Dosovitskiy et al., ICLR 2021.` at the end of the section. The full reference goes under *Further reading*.
- **Boxes:** GitHub alerts render on both GitHub and the website:

  ```markdown
  > [!TIP]
  > The key idea in one sentence.
  ```

  Use `NOTE`, `TIP`, `IMPORTANT` or `WARNING`.

## Notebooks

- The first cell is the title with the "Open in Colab" badge, as in the placeholders.
- Save notebooks **without outputs** (*Edit → Clear outputs of all cells* before committing): the website doesn't show them, and outputs make the repository large.
- Notebooks run in Colab, so use packages that Colab has, or install them in the first code cell with `%pip install ...`.

## Preview

From the repository root:

```bash
pip install -r docs-requirements.txt
mkdocs serve
```

Then open http://127.0.0.1:8000. The page reloads when you save a file.
