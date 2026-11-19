# Setup

Before you start, install the tools below and create the accounts you need. This takes about an hour, so do it in the first week.

## Tools and accounts

| Tool | What you use it for | Get it |
| --- | --- | --- |
| **VS Code** | Editing code, the terminal, and editing files on Snellius with the *Remote - SSH* extension | [code.visualstudio.com](https://code.visualstudio.com/download) |
| **Git** + a **GitHub** account | Version control; you hand in your code as a public GitHub repository | [git-scm.com](https://git-scm.com/downloads), [github.com/join](https://github.com/join) |
| **Weights & Biases** account | Following your training runs, which run on the cluster without a screen | [wandb.ai](https://wandb.ai/) |
| **MobaXterm** (Windows, optional) | SSH and drag-and-drop file transfer to Snellius. On macOS or Linux, `ssh` in a terminal is enough | [mobaxterm.mobatek.net](https://mobaxterm.mobatek.net/) |
| **Docker Desktop** | Building the submission image | [docker.com](https://www.docker.com/products/docker-desktop/) |

Useful VS Code extensions: **Python**, **Remote - SSH**, and **GitHub Pull Requests**.

## Fork and clone the repository

You work with two repositories:

- **your fork**, where you push your own work (`origin`),
- **the course repository**, where you pull our updates from (`upstream`).

1. Open the [course repository](https://github.com/TUE-ARIA/NNCV) and click **Fork**.
2. Clone **your fork**:
    ```bash
    git clone https://github.com/<your-username>/NNCV.git
    cd NNCV
    git remote add upstream https://github.com/TUE-ARIA/NNCV.git   # if `git remote -v` doesn't list it yet
    ```
3. Commit often and push only to your fork:
    ```bash
    git commit -m "Describe your change"
    git push origin main
    ```

We may push fixes to the course repository during the assignment. To get them:

```bash
git fetch upstream
git rebase upstream/main
git push --force-with-lease origin main
```

Don't push to the course repository, and don't remove the `upstream` remote.

> [!NOTE]
> The `Final assignment` folder name contains a space. Put quotes around paths that include it, e.g. `cd "NNCV/Final assignment"`.

## Check your setup

Run these in a terminal:

```bash
git --version
docker --version
docker run --rm hello-world     # Docker Desktop must be running
```

Also check that:

- VS Code opens your clone of the repository,
- you can sign in to GitHub and Weights & Biases,
- you can log in to Snellius over SSH (`ssh <username>@snellius.surf.nl`, or a new SSH session in MobaXterm).

## Tips

- **GitHub on Snellius:** create a Personal Access Token (*Settings → Developer settings → Personal access tokens*) and use it to clone, as described in [Snellius](snellius.md).
- **W&B API key:** you find it in your W&B account settings. You put it in `.env` on Snellius (see [Snellius](snellius.md)).
- **Debugging:** use breakpoints in VS Code ([how-to](https://code.visualstudio.com/docs/python/debugging)) on a small piece of data before you start long jobs.

Next: [Snellius](snellius.md), where you train your first baseline.
