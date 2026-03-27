from AntiAtropos.server.AntiAtropos_environment import AntiAtroposEnvironment
from AntiAtropos.models import SREAction, ActionType

def run_diagnostic(task_id, steps=30):
    print(f"\n{'='*20} DIAGNOSING: {task_id} {'='*20}")
    env = AntiAtroposEnvironment()
    obs = env.reset(task_id=task_id)
    
    # We'll track node-0 (Payment Gateway) and node-1 (Standard Node)
    print(f"{'Step':<5} | {'Node-0 Rate':<12} | {'Node-1 Rate':<12} | {'Total Queue':<12} | {'Notes'}")
    print("-" * 75)

    for i in range(1, steps + 1):
        # We issue NO_OP to see the "natural" evolution of the traffic
        action = SREAction(action_type=ActionType.NO_OP)
        obs = env.step(action)
        
        n0 = next(n for n in obs.nodes if n.node_id == "node-0")
        n1 = next(n for n in obs.nodes if n.node_id == "node-1")
        
        note = ""
        if task_id == "task-2" and i == 25:
            note = "<< FAILURE TRIGGERED >>"
        elif task_id == "task-3":
            if n1.incoming_request_rate > 100:
                note = "<< SURGE DETECTED >>"

        print(f"{i:<5} | {n0.incoming_request_rate:<12.1f} | {n1.incoming_request_rate:<12.1f} | "
              f"{obs.total_queue_backlog:<12} | {note}")

if __name__ == "__main__":
    # Test Task 1: Linear Ramp
    run_diagnostic("task-1", steps=15)
    
    # Test Task 2: Fault Tolerance (Failure at step 25)
    run_diagnostic("task-2", steps=26)
    
    # Test Task 3: Stochastic Surge
    run_diagnostic("task-3", steps=20)