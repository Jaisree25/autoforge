# AutoForge — Brev cloud setup

One-page bootstrap for running AutoForge on a Brev L40S box for the Hack-a-Claw demo.

---

## 0. Pick a GPU

| Option | VRAM | ~Cost | Notes |
|---|---|---|---|
| **L40S** ⭐ | 48 GB | ~$1.50-2.50/hr | Matches the envelope stub + the NVIDIA-flagship narrative. Recommended. |
| L4 | 24 GB | ~$0.80/hr | Solid cheaper alternative. |
| T4 | 16 GB | ~$0.35/hr | Plenty for MNIST/CSV; cheapest sane choice. |
| A10G | 24 GB | ~$1.00/hr | Fine if L40S unavailable. |

Base image: any **CUDA-enabled Ubuntu** template (22.04 or 24.04). Brev's
default PyTorch image is fine; we install our own conda env on top.

---

## 1. Push your local repo to GitHub

On your **local Windows box** before SSHing into Brev:

```powershell
# Verify .env is excluded (it must NOT appear)
git status --ignored | Select-String '\.env'

# First time only
cd D:\autoforge
git init
git add .
git status   # double-check no .env, no .pkl, no data/artifacts/* files

git commit -m "AutoForge MVP — agentic ML pipeline"
gh repo create autoforge --private --source=. --remote=origin --push
# or, manually:
#   git remote add origin git@github.com:<you>/autoforge.git
#   git branch -M main
#   git push -u origin main
```

**Never push `.env`** — your NVIDIA + Tavily keys live there. The
`.gitignore` excludes it, but verify with `git status` before every push.

---

## 2. SSH into Brev

Use Brev's web console or the `brev` CLI to get an SSH command, then:

```bash
ssh <brev-host>
```

Once inside, everything below runs on the Brev box.

---

## 3. System deps

```bash
sudo apt update
sudo apt install -y git build-essential curl
```

---

## 4. Miniconda (skip if already installed)

```bash
which conda && echo "conda already here" || {
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
  echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
}
```

---

## 5. Clone the repo

```bash
git clone git@github.com:<you>/autoforge.git
cd autoforge
chmod +x tasks.sh
```

---

## 6. Conda env

```bash
./tasks.sh setup
conda activate autoforge
```

This creates the `autoforge` env from `environment.yml`. Takes ~3-5 minutes.

---

## 7. `.env` — paste keys from your local notes

```bash
cp .env.example .env
nano .env   # or vim
```

Paste your `NVIDIA_API_KEY` and `TAVILY_API_KEY`. Leave the Telegram
fields blank unless you've wired the bot. Save and exit.

Verify it loaded:
```bash
./tasks.sh test-nemotron
# should print model + version response
```

---

## 8. Sanity tests

```bash
./tasks.sh smoke              # 38 tests, ~15s
./tasks.sh init-db            # creates data/autoforge.db
python scripts/create_mnist_fixture.py   # downloads + subsamples MNIST
```

---

## 9. Launch the dashboard

```bash
./tasks.sh dashboard
```

Streamlit serves on `localhost:8501`. To view from your laptop, either:

**Option A — port forwarding (recommended):**
```bash
# on your laptop, in a separate terminal:
ssh -L 8501:localhost:8501 <brev-host>
```
Then open http://localhost:8501 in your browser.

**Option B — Brev's public URL:**
Check the Brev console for the auto-forwarded port. Brev usually exposes
8501 with a public HTTPS URL.

---

## 10. Install Claude Code on the Brev box (optional but recommended)

If you want Claude Code on the box for in-place edits:

```bash
# Official installer (recommended)
curl -fsSL https://claude.ai/install.sh | bash

# Then auth:
claude login
# (opens a browser flow; on a headless box, follow the device-code prompt)
```

---

## 11. When NemoClaw drops

Check the Hack-a-Claw Discord / kickoff materials for the install line.
When you have it:

```bash
# Edit environment.yml — uncomment ONLY the nemoclaw line (leave openclaw commented):
#   - nemoclaw
nano environment.yml

# Also uncomment in requirements.txt if you use pip-only installs:
nano requirements.txt

# Re-sync the env:
./tasks.sh setup
```

Then swap `agents/_llm_client.py` one method at a time — the wrapper
already isolates every LLM call there. Agent code doesn't need to change.

---

## Common gotchas

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: config` | You're not in the project root. `cd ~/autoforge` first. |
| Streamlit shows "no module named X" | `conda activate autoforge` again — Brev shells sometimes start in base. |
| `ImportError: libcuda.so.1` | The conda env lacks GPU bindings; we run sklearn-CPU only today. Safe to ignore for the current demo. |
| `git pull` overwrites your `.env` | It won't — `.env` is gitignored. But back it up before any `git reset --hard`. |
| Port 8501 already in use | `lsof -ti:8501 \| xargs -r kill -9` then relaunch. |
| Claude Code can't auth on headless box | Use `claude login --no-browser` and paste the device code on your laptop. |

---

## Cost watch

L40S at ~$2/hr × 24 hr hackathon = ~$48. Stop the box when you're not
actively running. Brev usually has a one-click "stop" that preserves the
disk so you can resume later.

```bash
# Quick way to confirm you're using GPU (sklearn path won't):
nvidia-smi
```
