# QLoRA + OpenEnv Training Architecture (HF-Centric)

## 🧠 SYSTEM OVERVIEW
You are building a QLoRA + reward-based training system fully inside Hugging Face infrastructure.

Core principles:
- GPU = compute only (ephemeral)
- Hub = source of truth (persistent)
- Training = reproducible + resumable
- Metrics = structured + queryable

---

## 🧱 ARCHITECTURE

### 1. Code Repository (Control Plane)
Stored on Hugging Face Hub

Contents:
- train.py
- openenv_loop.py
- model_utils.py
- eval.py
- requirements.txt
- config.yaml

---

### 2. Training Runtime (Execution Plane)
Runs via Hugging Face Training Jobs

- GPU: A10G recommended
- Pulls code from Hub
- Uses temporary disk
- Must explicitly persist outputs

---

### 3. Model Storage
Stored on Hugging Face Hub

Contents:
- LoRA adapter weights
- Optional merged model

---

### 4. Metrics Storage
Stored as dataset on Hugging Face Hub

Example schema:
{
  "run_id": "exp_001",
  "step": 120,
  "train_reward": 0.82,
  "eval_reward_base": 0.65,
  "eval_reward_ft": 0.78,
  "loss": 1.23
}

---

### 5. Optional Artifacts
- plots (.png)
- logs
- evaluation outputs

---

## 🔁 TRAINING PROCESS FLOW

### Step 1 — Initialization
- Load base model (4-bit)
- Attach LoRA adapters
- Load tokenizer
- Resume from checkpoint if exists

---

### Step 2 — Training Loop
generate → evaluate → reward → update → log → checkpoint

---

### Step 3 — Logging
- Buffer metrics locally
- Push every N steps to dataset

---

### Step 4 — Evaluation
- Evaluate base vs fine-tuned
- Log results

---

### Step 5 — Checkpointing
- Save locally
- Push to Hub
- Ensure resumability

---

### Step 6 — Visualization
- Generate plots
- Save + push

---

## 🔄 RESUME LOGIC
- Check latest checkpoint
- Resume if exists
- Else start fresh

---

## 📊 POST-TRAINING
- Load dataset
- Plot reward curves
- Compare models

---

## ⚙️ CONFIG
- batch_size
- learning_rate
- lora_rank
- eval_interval
- checkpoint_interval
- push_interval
- run_id

---

## ⚠️ FAILURE HANDLING
- Frequent checkpointing
- Frequent metric pushes

---

## 🧠 FINAL MODEL
HF Hub = database  
HF Jobs = compute  
QLoRA = training layer  
OpenEnv = reward  
Datasets = tracking  

---

## ✅ SUCCESS CRITERIA
- Resumable training
- Persistent metrics
- Comparable runs
- No data loss