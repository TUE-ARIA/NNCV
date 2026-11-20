// MathJax for the course website: renders the formulas that pymdownx.arithmatex marks.
// The site loads pages without a full reload (navigation.instant), so the formulas are
// typeset again after every page change, not only on the first load.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

let firstPage = true;
document$.subscribe(() => {
  if (firstPage) {             // MathJax typesets the first page itself when it loads
    firstPage = false;
    return;
  }
  const mj = window.MathJax;
  if (!mj.startup || !mj.startup.promise) return;
  // Chain onto the previous typeset, so two runs never process the same formulas at once.
  mj.startup.promise = mj.startup.promise.then(() => {
    mj.startup.output.clearCache();
    mj.typesetClear();
    mj.texReset();
    return mj.typesetPromise();
  });
});
