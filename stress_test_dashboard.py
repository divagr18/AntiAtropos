import asyncio
import os
import time
from AntiAtropos.client import AntiAtroposEnv
from AntiAtropos.models import SREAction, ActionType

ENV_URL = os.getenv("ANTIATROPOS_ENV_URL", "https://pranavkk-antiatropos.hf.space")

async def run_stress_test():
    print(f"Connecting to {ENV_URL}...")
    async with AntiAtroposEnv(base_url=ENV_URL) as env:
        
        # --- TEST 1: Task-1 (Scaling Burst) ---
        print("\n[Test 1] Starting Task-1 Burst Scaling...")
        await env.reset(task_id="task-1", mode="simulated")
        
        for i in range(10):
            action = SREAction(action_type=ActionType.SCALE_UP, target_node_id=f"node-{i%5}", parameter=2.0)
            res = await env.step(action)
            print(f"  Step {i+1}: SCALE_UP on node-{i%5} | Reward: {res.reward:.4f}")
            # No sleep = high throughput
        
        # --- TEST 2: Task-3 (Invalid Actions & Shenanigans) ---
        print("\n[Test 2] Starting Task-3 Invalid Actions (SLA Breaches)...")
        await env.reset(task_id="task-3", mode="simulated")
        
        # Issue a forbidden SHED_LOAD on node-0 (VIP) to trigger metadata/errors
        for i in range(5):
            # SHED_LOAD on VIP should trigger rejection/SLA issues in Task-3 logic
            action = SREAction(action_type=ActionType.SHED_LOAD, target_node_id="node-0", parameter=1.0)
            res = await env.step(action)
            ack = res.observation.action_ack_status
            print(f"  Step {i+1}: SHED_LOAD on node-0 | Ack: {ack} | Score: {res.reward:.4f}")
            time.sleep(0.5) # Slight delay to see the step progression

        print("\nStress test complete! Check Grafana for:")
        print("1. 'Action Throughput' spike for SCALE_UP and REROUTE.")
        print("2. 'Task' filter dropdown should now show task-1 and task-3.")
        print("3. 'SLA Violations' should have incremented from Test 2.")
        print("4. 'Node Status Table' should reflect the queue/latency from the last steps.")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
