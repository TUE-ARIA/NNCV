# Submission Guidelines: Writing a Great Report  

Hi there! 🎉 Congratulations on making it to the final stages of this course. Writing a strong report is just as important as training a great model. Your report is the primary way we evaluate your work, so let’s make sure it reflects all the effort you’ve put into this project. Below are some do’s and don’ts to help you write a publication-grade report for your final submission.  

---

## Do’s ✅  

### 1. **Use Test Set Results**  
- Always report results from the **test set** as evaluated by the Challenge submission server.  
- Validation results from your own train-validation split are useful for debugging and hyperparameter tuning, but they are **not** a substitute for test set results.
- Simply stating that *"the model was not converged"* as a reason for underperformance is **not good enough**. We provide sufficient compute for everyone to train their models properly. Make sure your models are fully trained and converged, and report the test set results from the model that gave optimal performance on your validation set.  

### 2. **Include Results for Multiple Benchmarks**  
- We expect to see results and baseline comparisons for at least **two benchmarks**: peak performance and one other.  
- Make sure to include both benchmarks in your report, along with methodological improvements, results, and discussions.  
- For clarity, it is advised to create separate subsections for the different benchmarks and their respective baselines. This helps structure your report and makes it easier to follow.  

### 3. **Be Clear and Structured**  
- Include the following sections:  
  - **Abstract**: A short summary of your problem, approach, and key findings.  
  - **Introduction**: Explain the problem, its challenges, and your motivation.  
  - **Methods**: Describe your dataset, baseline model, and any improvements you made.  
  - **Results**: Present your findings with clear tables, graphs, and explanations. Include qualitative examples (e.g., segmentation outputs) to help convey your model’s performance effectively.
  - **Discussion**: Reflect on your results, limitations, and potential future work.  

### 4. **Follow the IEEE Format**  
- Use the [IEEE double-column format](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn) for your report.  
- Keep it concise: **3-4 pages** is the target length.  
- We strongly advise you to use all 4 pages to provide as much detail as possible. However, do not exceed 4 pages, as any text (excluding references) beyond this limit will **not** be included for grading.  

### 5. **Use Visuals Effectively**  
- Include **figures** and **tables** to support your findings.  
- Label all visuals clearly and reference them in the text (e.g., "As shown in Figure 1...").  
- Nice visuals really help in understanding what you did and can give a great overall impression of your paper. Spend some time making them clear and professional.  
- Make sure the text inside figures and tables is large enough to be easily readable (no smaller than 2 font sizes below the default text font size).  
- In a double-column format, you can use the `figure*` environment in LaTeX to spread an image across both columns if needed.  

### 6. **Cite Your Sources**  
- If you use ideas, methods, or code from other papers or repositories, **cite them properly**.  
- Use a consistent citation style (e.g., IEEE).  
- Don’t forget to add a reference to your code (e.g., GitHub repository) in the paper. Usually, this is done at the end of the **Introduction** or in the **Methods** section. For example:  
  *"The code used to produce the results presented in this paper is publicly available at [https://www.github.com/.../...](https://www.github.com/.../...)."* Easy points! 😃  

### 7. **Proofread Your Report**  
- Check for grammar, spelling, and formatting errors.  
- It’s fine to use tools like GenAI (e.g., ChatGPT) to check your grammar and spelling, but a report written by GenAI will be considered **plagiarism**.  
- Ask a peer to review your report for clarity and coherence.  

### 8. **Include a Clear README in Your Repository**  
- Make sure your submitted repository includes a well-written **README** file.  
- The README should clearly explain how to reproduce your results, including:  
  - Installation instructions for dependencies.  
  - Steps to preprocess the data.  
  - Commands to train and evaluate your model.  
  - Any additional details required to replicate your findings.  
- A clear and detailed README makes it easier for us to evaluate your work and ensures reproducibility.  

---

## Don’ts ❌  

### 1. **Don’t Use Validation Results as Final Results**  
- Reporting validation results instead of test set results is a common mistake. The test set is the **only** reliable benchmark for your model’s performance.  

### 2. **Don’t Ignore the Baseline**  
- Always compare your results to the provided baseline model. This helps us understand the impact of your improvements. The code for training the baseline has already been provided to you, so 
again, easy points! 😃  

### 3. **Don’t Use Low-Quality Visuals**  
- Avoid snipping ugly graphs from tools like Weights & Biases or tables from Excel. These can make your report look unprofessional. Instead, spend some time creating clean, high-quality visuals that enhance your paper.  
- Loss curves often take up a lot of space and are usually not very relevant to your results. Only include them if they are essential for your discussion (e.g., to demonstrate convergence or other critical insights).  

![](img/table_example.png)

### 4. **Don’t Overload with Technical Terminology**  
- Write for a general audience with a technical background. Avoid unnecessary complexity.  

### 5. **Don’t Skip the Discussion Section**  
- A good report doesn’t just present results; it reflects on them. Discuss what worked, what didn’t, and why.  

### 6. **Don’t Plagiarize**  
- Plagiarism will result in a **zero** for your report. Always give credit where it’s due.  

---

## Final Tips 💡  

- **Start Early**: Writing a good report takes time. Don’t leave it for the last minute!  
- **Ask for Help**: If you’re unsure about something, reach out to us or your peers. Use the **Discussions** section of the repository to collaborate.  
- **Be Honest**: If your model didn’t perform as expected, that’s okay! A well-analyzed failure can be just as valuable as a success.  

We’re excited to read your reports and see the amazing work you’ve done. Good luck, and happy writing! 🚀  

---