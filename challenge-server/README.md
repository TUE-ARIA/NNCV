# Challenge server

For staff. The website where students submit to the final-assignment challenge, plus the worker that runs their submissions on the server's GPU and scores them. Students find the instructions on the site itself, under *How to submit*, and in the course documentation.

```text
challenge-server/
├── config.yaml        all settings: quotas, deadline, hidden test, limits, base images
├── compose.yaml       starts the website (web) and the job runner (worker)
├── manage.py          staff commands: students, baselines, queue
└── app/               web.py (website), worker.py (runs jobs), sandbox.py (Docker settings),
                       rules.py (quotas, ranking), packages.py (upload checks), templates/, static/
```

## How it works

1. A student chooses their name, enters their password, and uploads a **zip** with their code and weights, made with `python -m submission.package`. They can also upload a **complete Docker image** (`docker save`).
2. The upload is checked right away, against the zip structure, quota, deadline and file size. It then waits in the queue: one job at a time, taking turns per student.
3. The **worker** runs the job on the GPU in a locked-down container:
    - no network, a read-only file system, a non-root user, and limits on memory and time;
    - only the *input* images mounted, never the ground truth;
    - the fixed harness (`final-assignment/submission/run.py`) from this repository, mounted read-only.

    Zips run on a base image from `config.yaml`. Those images are built from `final-assignment/docker/` and published by the *Base images* workflow.
4. A separate, trusted **scorer** container runs `final-assignment/evaluate.py` on the predictions. The scores appear on the leaderboard of the track.

The metrics and harness are the exact files that students have in their fork, so local and server scores match.

## First-time setup

On the server (Ubuntu with Docker, the NVIDIA container toolkit, and a working `nvidia-smi`):

```bash
sudo mkdir -p /srv/nncv && sudo chown $USER /srv/nncv
git clone https://github.com/TUE-ARIA/NNCV.git /srv/nncv/repo        # the harness is read from here
cd /srv/nncv/repo/challenge-server
cp .env.example .env && nano .env                                     # set NNCV_SECRET

# evaluation data in the server layout (see final-assignment/tools/make_eval_sets.py)
#   /srv/nncv/eval/test/      ATLAS test split: the validation leaderboard
#   /srv/nncv/eval/hidden/    the hidden test set
#   /srv/nncv/eval/reference_segmenter.pth

docker compose up -d --build
docker compose exec worker python manage.py pull-images         # base images + scorer
docker compose exec worker python manage.py prefetch-weights    # FID/FVD networks for the offline scorer
```

The site then runs on port 80 of the server (change `NNCV_PORT` in `.env`).

> [!NOTE]
> The base images on the GitHub Container Registry are private when first published. Make them public once, under the organization's *Packages* → package → *Package settings* → *Change visibility*, so students and the server can pull them without logging in.

## Starting and stopping

Run these from `/srv/nncv/repo/challenge-server`:

| Task | Command |
| --- | --- |
| Shut down | `docker compose down` |
| Start | `docker compose up -d` |
| Restart | `docker compose restart` |
| Update after `git pull` | `docker compose up -d --build` |
| Is it running? | `docker compose ps` |
| Follow the jobs | `docker compose logs -f worker` (Ctrl+C stops following, not the server) |

After a reboot of the server the site and worker start by themselves, unless they were shut down with `docker compose down`. Students, submissions and results are kept in `/srv/nncv/data`, outside the containers. A job that was running goes back into the queue and runs again from the start.

To stop new submissions without shutting down, set `submissions.open: false` in `config.yaml`. The leaderboards stay online.

## Accounts: students, staff and baselines

```bash
docker compose exec worker python manage.py import-students roster.csv passwords.csv
docker compose exec worker python manage.py add-baseline "Baseline"
docker compose exec worker python manage.py submit "Baseline" consistency baseline_consistency.zip --note "U-Net + ConvGRU"
```

The roster is a CSV with a header row: `name`, and optionally `email` and `role` (`student`, the default, or `staff`):

```csv
name,email,role
Jane Doe,j.doe@tue.nl,staff
John Smith,j.smith@student.tue.nl,
```

`import-students` generates a password per account and writes them to `passwords.csv`. Send everyone their password, then delete the file: the server only stores hashes. Put the roster and password files in `/srv/nncv` so the container can read them. Emails are stored but never shown on the site.

**Staff** sign in on the site like students and get an *Admin* page with all submissions (filter by status, track and student), every log including the scorer log, and the hidden-test scores before they are revealed. From there they can cancel waiting jobs, run jobs again, give an account a new password and deactivate accounts. Staff submissions have no quota and are never ranked. Change a role with `manage.py set-role "Name" staff` (or `student`). Settings stay in `config.yaml`, for whoever has access to the server.

To clear test accounts and results before the course starts:

```bash
docker compose down
sudo rm -rf /srv/nncv/data/nncv.sqlite* /srv/nncv/data/uploads /srv/nncv/data/jobs
docker compose up -d
```

## The course folder on Snellius

Students' `slurm/setup.sh` links everything from the shared folder `SHARED_DIR` in `final-assignment/slurm/servers/snellius.env`. Fill it once, and give students read access:

```text
<SHARED_DIR>/
├── atlas120k/                  the dataset: train / val / test
├── eval/val/, eval/test/       made with tools/make_eval_sets.py, as on the server
├── reference_segmenter.pth     the ATLAS reference model
└── containers/                 the three base images
```

```bash
cd <SHARED_DIR>/containers
for image in pytorch transformers generative; do
  apptainer pull --force nncv-$image.sif docker://ghcr.io/tue-aria/nncv-$image:2027
done
```

Run the same loop after the images are rebuilt. Students' links then use the new images at once. Without `containers/`, `setup.sh` downloads the `pytorch` image for every student into their own folder, which costs each of them several GB of quota.

## During the course

| Task | How |
| --- | --- |
| Open the hidden test (final week) | `test_sets.hidden.enabled: true` in `config.yaml` |
| Reveal hidden-test scores (challenge day) | `test_sets.hidden.show_scores: true` |
| Set the deadline | `submissions.deadline: "2027-04-02 23:59"` |
| Change quotas, limits, time limits | `config.yaml`; takes effect immediately |
| See all submissions | the *Admin* page, or `python manage.py list` |
| Cancel or rerun a job | the *Admin* page, or `python manage.py cancel ID`, `python manage.py requeue ID` |
| New password for an account | the *Admin* page, or `python manage.py reset-password "Name"` |
| Logs of a failed job | *log* on the *Admin* page, or `/srv/nncv/data/jobs/<id>/inference.log` and `score.log` |
| Update the harness or metrics | `git pull` in `/srv/nncv/repo`; the next job uses it |
| Update the website or worker | `git pull`, then `docker compose up -d --build` |
| Restart after changing `paths` | `docker compose restart` |

Run `manage.py` inside the worker container: `docker compose exec worker python manage.py <command>`. `python manage.py --help` lists all commands.

Back up `/srv/nncv/data/nncv.sqlite` regularly; it holds all students and results.

## Security notes

- The site is meant for the TU/e network only and runs over plain HTTP. Passwords are generated by the server, so students don't reuse their own.
- Staff accounts see hidden-test scores and every log, so give the role to staff only. Their sign-in lasts 8 hours.
- Student code never has network access, and never sees the ground truth or the scoring code's results.
- Only the worker has access to Docker. It needs the Docker socket to start jobs, which makes it equivalent to root on this server: keep the server dedicated to the challenge.
- The FPS of the Real-time track is measured inside the job, where student code runs in the same process and could in principle tamper with the timer. Every job's total run time is recorded too, and the uploaded code is kept for review: check the top Real-time entries before the final ranking.
