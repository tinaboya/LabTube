# Running the pipeline on Google Cloud (or any VM with more RAM)

Use this when your laptop doesn’t have enough memory for the full lab-events → admissions step (~158M rows).

---

## 1. Which machine to choose (Google Cloud)

- **RAM:** At least **16 GB**, ideally **32 GB**. The admission step (DuckDB join/sort/backfill) needs this; 32 GB is comfortable.
- **vCPUs:** **4** is enough; **8** is plenty.
- **Disk:** Enough for your data: `labevents.parquet` + `admissions.parquet` + output. Plan for **50–100 GB** free (or more if you keep CSVs).

**Concrete options on GCP:**

| Option | Machine type | RAM | vCPUs | When to use |
|--------|--------------|-----|-------|-------------|
| **Cheapest** | **e2-standard-8** (General purpose) | 32 GB | 8 | Best value, usually enough |
| **Same, smaller** | **e2-standard-4** | 16 GB | 4 | If 32 GB isn’t available / you want to save cost |
| **Memory-optimized** | **M3** series, smallest with ≥32 GB RAM | 32+ GB | 4–8 | If e2 isn’t available or you want more RAM headroom |

In the GCP Console: **Compute Engine → VM instances → Create instance**. Pick a **region** (e.g. same as your group), then under **Machine configuration** choose **e2** or **Memory-optimized (M3)** and pick a type with **32 GB RAM** (e.g. e2-standard-8). Set **Boot disk** to e.g. **Ubuntu 22.04**, size **50–100 GB**. Allow **HTTP/HTTPS** if you need it; for SSH-only you don’t need to open extra ports.

---

## 2. Create the VM and get SSH access

1. In **Google Cloud Console**: Compute Engine → **VM instances** → **Create instance**.
2. Set **Name**, **Region**, **Machine type** (e.g. e2-standard-8), **Boot disk** (Ubuntu 22.04, 50–100 GB).
3. Under **Security** (or **Access**): either add your **SSH key** (recommended) or use “**Set up OS Login**” and add your Google account for SSH.
4. Click **Create**. Note the **External IP** (e.g. `34.xxx.xxx.xxx`).

---

## 3. Connect from your laptop (SSH)

From your **Mac terminal** (not on the VM):

```bash
# If you use an SSH key (e.g. added in GCP metadata):
ssh -i ~/.ssh/your-private-key your_username@EXTERNAL_IP

# If you use gcloud and OS Login (after: gcloud auth login, gcloud config set project YOUR_PROJECT_ID):
gcloud compute ssh INSTANCE_NAME --zone=ZONE
```

Replace `your_username` with the user GCP shows (often your local username or a Google-managed one). If you’re unsure, GCP shows the exact `ssh` command in the VM list: click **SSH** next to the instance and use “**View gcloud command**” or the command it prints.

---

## 4. Copy the project and data to the VM

From your **laptop** (same machine where your Latte folder lives):

```bash
# Replace with your VM’s IP and username
VM="your_username@EXTERNAL_IP"
LATTE="/Users/boya_unige/Documents/MIT-projects/Latte/Latte"

# Copy the whole project (scripts + MIMIC4 data). Use rsync so you can resume if it drops.
rsync -avz --progress -e "ssh" "$LATTE" "$VM:~/"

# If rsync isn’t available, use scp:
scp -r "$LATTE" "$VM:~/"
```

This puts a `Latte` folder in the VM’s home directory. Ensure `MIMIC4/labevents.parquet` and `MIMIC4/admissions.parquet` (and `d_labitems.csv` if you use vocab) are inside that folder before copying.

---

## 5. On the VM: install dependencies and run

SSH into the VM (step 3), then:

```bash
cd ~/Latte

# Install Python 3 and pip if needed (Ubuntu)
sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv

# Option A: use uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python scripts/lab_with_admissions.py

# Option B: plain pip
python3 -m venv .venv
source .venv/bin/activate
pip install duckdb polars pyarrow
python scripts/lab_with_admissions.py
```

The script reads `MIMIC4/labevents.parquet` and `MIMIC4/admissions.parquet` and writes `MIMIC4/lab_events_with_adm.parquet`. With 32 GB RAM you can leave the default DuckDB settings (or set `DUCKDB_MEMORY_LIMIT=16GB` if you want).

---

## 6. Copy the result back to your laptop (optional)

From your **laptop**:

```bash
VM="your_username@EXTERNAL_IP"
scp "$VM:~/Latte/MIMIC4/lab_events_with_adm.parquet" /Users/boya_unige/Documents/MIT-projects/Latte/Latte/MIMIC4/
```

Or use `rsync` to sync the whole `MIMIC4` folder.

---

## 7. Shut down the VM when done

In **Google Cloud Console**: Compute Engine → VM instances → select the instance → **Stop**. You’re charged mainly while it’s running; stopping it stops the billing for the VM (you may still pay a little for the disk until you delete it).

---

## Summary

| Step | Where | Action |
|------|--------|--------|
| 1 | GCP Console | Create VM: e2-standard-8 (32 GB RAM) or M3 with 32 GB, Ubuntu, 50–100 GB disk |
| 2 | GCP Console | Note External IP, set up SSH key or OS Login |
| 3 | Your Mac terminal | `ssh your_user@EXTERNAL_IP` (or `gcloud compute ssh ...`) |
| 4 | Your Mac terminal | `rsync -avz --progress -e "ssh" /path/to/Latte user@IP:~/` |
| 5 | VM (after SSH) | `cd ~/Latte`, `uv sync` (or pip install), `uv run python scripts/lab_with_admissions.py` |
| 6 | Your Mac terminal | `scp user@IP:~/Latte/MIMIC4/lab_events_with_adm.parquet ./MIMIC4/` |
| 7 | GCP Console | Stop (or delete) the VM |
