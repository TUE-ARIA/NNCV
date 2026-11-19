# Report, pitch & grading

The final assignment is **50% of your course grade**. It is graded on three things: your **code repository**, your **paper** and your **pitch** on the challenge day. This page explains each of them.

## Deliverables

| What | Format | Where |
| --- | --- | --- |
| Paper | 3-4 pages, [IEEE conference template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn), PDF | Canvas |
| Code | Your public fork of the course repository, with a README (see below) | Link in the paper and on Canvas |
| Pitch | 2 minutes, at most 4 slides | Challenge day |
| Challenge submissions | Dice track plus at least one other track, on the validation and the hidden test servers | Challenge servers ([Submitting](submission.md)) |

Text beyond 4 pages is not graded; references don't count towards the limit.

The README of your repository explains how to reproduce your results: the extra packages you use, the exact `slurm/submit.sh` commands for your final models, and your challenge-server username and TU/e email address.

## Grading

| Criterion | What we look for |
| --- | --- |
| **Clarity** | A reader or listener who knows deep learning but not your project can follow what you did and why. Figures and tables are readable and referred to. |
| **Experimental design** | Every change is compared with the baseline of its track, one change at a time where possible, with all metrics of the track. |
| **Insight** | You explain *why* something works or fails, with evidence: ablations, failure cases, per-class or per-procedure results. |
| **Originality** | Ideas that go beyond tuning the baseline, motivated by the problem or the literature. |
| **Reproducibility** | Your repository runs, and its README reproduces your results. |

A well-analysed method that doesn't rank highly can still get a 10.

## The challenge day

In the final week, everyone presents their work at the challenge day, and everyone attends the whole event, so you see what the others did.

- **Your pitch:** 2 minutes and at most 4 slides, about your **methods and results**, with some discussion if it fits. Skip the introduction to the task: everyone in the room knows it.
- **The final ranking:** the hidden test results are announced at the end of the day, per track. No one sees them before then.
- **Bonus points**, based on that ranking and added to your final assignment grade: **+0.5** for the best result in a track, **+0.25** for a top-3 result. Bonuses add up over tracks.

## Writing the paper

### Structure

| Section | Content |
| --- | --- |
| Abstract | Problem, approach and main result in 100-300 words. |
| Introduction | The clinical problem, why it is hard, related work, your contribution, and a link to your code. |
| Methods | The data and preprocessing, the baseline of each track you took part in, and what you changed. |
| Results | Validation-server results per track, compared with that track's baseline, plus qualitative examples. |
| Discussion | What worked and why, what didn't, limitations, and what you would try next. |

Use one subsection per track in Methods and Results.

### Results

- **Report the validation-server scores** (the `test` split) as your main results. Use `val` to choose your models and settings; the validation server is for checking the final ones, not for tuning.
- **Report every metric of a track**, not only the ranked one: Dice and Change Dice for Consistency, Dice and FPS for Real-time, FVD, FID and mask Dice for Generative.
- **Train to convergence.** "The model had not converged" is not an explanation; there is enough compute for everyone.
- **Show examples**: predicted masks next to the ground truth, generated frames next to real ones, or a failure case you can explain.

A results table puts the baseline first and adds one change per row. With the `booktabs` package:

```latex
\begin{table}[t]
  \centering
  \caption{Consistency track, validation server (test split). Best result in bold.}
  \label{tab:consistency}
  \begin{tabular}{lccc}
    \toprule
    Method                         & Dice $\uparrow$ & Change Dice $\uparrow$ & Score $\uparrow$ \\
    \midrule
    Baseline (U-Net + ConvGRU)     & 0.512           & 0.301                  & 0.407            \\
    + temporal loss                & 0.508           & 0.344                  & 0.426            \\
    + flow-warped memory           & \textbf{0.519}  & \textbf{0.362}         & \textbf{0.441}   \\
    \bottomrule
  \end{tabular}
\end{table}
```

The numbers are made up. Use horizontal rules only, state in the caption which split the numbers come from, and mark whether higher or lower is better.

### Figures

- Make figures for the paper, not screenshots of W&B or a spreadsheet. Text in a figure is at most two sizes smaller than the body text.
- Use `figure*` for a figure across both columns.
- Leave out loss curves unless your argument needs them.

### Sources, GenAI and plagiarism

- Cite every paper, method and repository you build on, in one consistent style (IEEE).
- You may use GenAI to check spelling and grammar. A paper written by GenAI counts as plagiarism.
- Plagiarism gives a zero for the report.

## Before you hand in

For the paper and code:

- [ ] Validation-server results for the Dice track and at least one other track, each compared with its baseline.
- [ ] All metrics of each track reported, with the split named in every table.
- [ ] At most 4 pages of text, in the IEEE template.
- [ ] A link to your public repository in the paper, and a README that reproduces your results.
- [ ] All sources cited.
- [ ] Read by a fellow student.

For the pitch:

- [ ] At most 4 slides, about methods and results, that fit in 2 minutes.
- [ ] Final models submitted to the hidden test servers before the deadline.

Next: [FAQ](faq.md).
