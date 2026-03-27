from AntiAtropos.server.AntiAtropos_environment import AntiAtroposEnvironment
from AntiAtropos.models import SREAction, ActionType

def test_lyapunov_spike():
    print(f"{'Step':<5} | {'Lyapunov Energy V(s)':<25} | {'Status'}")
    print("-" * 50)
    
    env = AntiAtroposEnvironment()
    obs = env.reset(task_id="task-2") # Fault Tolerance Task
    
    # Run for 40 steps to see the failure at step 25 and the aftermath
    for i in range(1, 41):
        # We take NO action so we can see the "natural" instability
        action = SREAction(action_type=ActionType.NO_OP)
        obs = env.step(action)
        
        status = ""
        if i == 24:
            status = "Pre-failure (Stable)"
        elif i == 25:
            status = "<< NODE FAILED >>"
        elif i > 25:
            status = "Post-failure (Unstable)"

        # Access the energy calculated by stability.py
        print(f"{obs.step:<5} | {obs.lyapunov_energy:<25.2f} | {status}")

if __name__ == "__main__":
    test_lyapunov_spike()