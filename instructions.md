This document serves as the **Software Requirements Specification (SRS)** for **AntiAtropos**, an autonomous infrastructure management environment built for the Meta PyTorch OpenEnv Hackathon.
https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard?utm_source=midfunnel&utm_medium=email&utm_campaign=r1_confirmation_team
---

## 1. Competition Context & Constraints
**Event:** Meta PyTorch OpenEnv Hackathon x SST (India AI Hackathon '26)
**Round 1 Goal:** Build a real-world utility OpenEnv environment for AI agents.

### **Core Requirements (The "Agent's Memory")**
* **No Toys/Games:** Must simulate a task humans (SREs/DevOps) actually perform.
* **OpenEnv Spec:** Implement `step()`, `reset()`, `state()`. Use typed Pydantic models for Action, Observation, and Reward.
* **Task Structure:** 3+ tasks (Easy → Medium → Hard) with deterministic graders (0.0–1.0).
* **Reward Function:** Must provide partial progress signals (not just binary success).
* **Baseline:** Reproducible inference script using the OpenAI API client.
* **Deployment:** Containerized (Dockerfile) and hosted on a **Hugging Face Space**.
* **Deadline:** April 7, 2026.

---

## 2. Project Vision: AntiAtropos
**Concept:** An autonomous Site Reliability Engineer (SRE) agent that manages a cluster of microservices. It balances operational cost against system stability using **Lyapunov-inspired control logic**.



---

## 3. Modular System Architecture
The system is designed in four swappable layers to allow for rapid iteration and "vibecoding" of infrastructure while maintaining mathematical rigour.

### **Layer A: The Infrastructure Simulator (Core)**
* **Module:** `simulator.py`
* **Function:** A discrete-time simulation of a microservice cluster. 
* **Logic:** Tracks `incoming_requests`, `node_health`, `queue_depth`, and `latency` for $N$ nodes.
* **Modularity:** Can be replaced with a more complex physics-based grid or a simple network graph without changing the API.

### **Layer B: The Stability & Lyapunov Layer (Math)**
* **Module:** `stability.py`
* **Function:** Calculates the **Lyapunov Energy** $V(s)$ and **Barrier Violations** $h(s)$.
* **Equation:** $V(s) = \sum Q_i^2$ (Sum of squared queue lengths).
* **Modularity:** Allows swapping between a "Hard Shield" (Rule-based) and a "Soft Lyapunov Penalty" (RL-based).

### **Layer C: The OpenEnv Interface (Bridge)**
* **Modules:** `models.py`, `client.py`, and `server/AntiAtropos_environment.py`
* **Function:** Maps simulator states to typed Pydantic models and runs the execution logic via a strict client-server separation as dictated by the OpenEnv standard.
* **Components:**
    * `models.py`: Strongly-typed Pydantic schemas (`Action`, `Observation`, `State`).
    * `server/AntiAtropos_environment.py`: The `Environment` subclass. Contains the actual `reset()` and `step()` server-side logic integrating `simulator.py` and `stability.py`.
    * `client.py`: The async/sync `EnvClient` wrapper used by AI models to interact over HTTP/WebSockets.
* **Core API (Handled via OpenEnv):**
    * `reset()`: Re-initializes the simulator and returns the initial `Observation`.
    * `step(action)`: Accepts `Action`, advances `simulator.py` by one tick, checks `stability.py`, and returns `Reward` & resulting `Observation`.

### **Layer D: The Grader & Evaluation Layer**
* **Module:** `grader.py`
* **Function:** Programmatic evaluation of agent performance. 
* **Metrics:** Uptime %, Cost efficiency, and "Stability Variance" (how well it kept the Lyapunov energy low).

---

## 4. Functional Specifications

### **4.1 Observation Space (The "Dashboard")**
A JSON object representing what a human SRE sees:
* `cluster_id`: String
* `active_nodes`: Integer
* `average_latency_ms`: Float
* `error_rate`: Float (0.0 - 1.0)
* `total_queue_backlog`: Integer
* `current_cost_per_hour`: Float

### **4.2 Action Space (The "CLI Tools")**
A Pydantic model representing the commands an agent can run:
* `action_type`: Enum [`SCALE_UP`, `SCALE_DOWN`, `REROUTE_TRAFFIC`, `SHED_LOAD`]
* `target_node_id`: String
* `parameter`: Float (e.g., number of nodes to add or % of traffic to shed)

### **4.3 The Lyapunov Reward Function**
$$R_t = -(\alpha \cdot \Delta V(s) + \beta \cdot \text{Cost} + \gamma \cdot \text{SLA\_violation\_step})$$
* Ensures the agent prioritizes **stability** ($\Delta V$) before **cost saving**.

---

## 5. Task Definitions (MVP for Round 1)

1.  **Task 1: Predictive Scaling (Easy)**
    * *Scenario:* Traffic increases linearly.
    * *Objective:* Scale nodes to keep latency $<200ms$ without over-provisioning.
2.  **Task 2: Fault Tolerance (Medium)**
    * *Scenario:* A random high-traffic node fails.
    * *Objective:* Detect failure and reroute traffic using `REROUTE_TRAFFIC` before queues explode.
3.  **Task 3: Stability Under Surge (Hard)**
    * *Scenario:* A massive, stochastic DDoS-style burst.
    * *Objective:* Use `SHED_LOAD` on non-critical endpoints to maintain Lyapunov stability for the VIP "Payment Gateway" (`node-0`), which carries higher business impact than the rest of the cluster.

---

## 6. Development Roadmap (Modular Slotting)

| Phase | Component | Focus |
| :--- | :--- | :--- |
| **Phase 1** | **Interface Definition** | Populate the scaffolded `models.py`, `client.py`, and `server/AntiAtropos_environment.py` with specific Pydantic schemas and logic definitions. |
| **Phase 2** | **Basic Simulator** | Build `simulator.py` with simple queue math. Inject it into the `AntiAtropos_environment.py` loop. |
| **Phase 3** | **Lyapunov Grader** | Implement `stability.py` to calculate drift and score the agent's performance. |
| **Phase 4** | **Baseline Agent & Local Testing** | Test locally via `uv run server`. Write `baseline.py` using an OpenAI/Groq client to test if the tasks are solvable in the environment. |
| **Phase 5** | **HF Deployment** | Configure `server/requirements.txt` and run `openenv push` to securely containerize and deploy the app to Hugging Face Spaces. |

---

Steps given on the Site
Step 1

 Scaffold
$
openenv init my_env
Copy
Generate project structure.

Step 2

Build
Define your environment in the generated files.

Step 3

Test locally
$
uv run server
Copy
Step 4

Deploy
$
openenv push --repo-id your-username/my-env
Copy
Step 5

 Submit
Paste your HF Spaces URL here before the deadline.
