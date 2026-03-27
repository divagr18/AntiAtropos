import time
from AntiAtropos.client import AntiAtroposEnv
from AntiAtropos.models import SREAction, ActionType, NodeStatus
from AntiAtropos.grader import grade_episode

def decide_action(obs) -> SREAction:
    """
    Heuristic logic to choose the most critical management action.
    """
    # 1. Identify nodes in trouble (approaching the 80-req threshold)
    #
    distressed_nodes = [
        n for n in obs.nodes 
        if n.status != NodeStatus.FAILED and n.queue_depth > 60
    ]
    
    if not distressed_nodes:
        return SREAction(action_type=ActionType.NO_OP)

    # 2. Prioritize the node with the largest backlog
    #
    worst_node = max(distressed_nodes, key=lambda n: n.queue_depth)
    
    # 3. Emergency Load Shedding
    # If queue is critical (>120) or latency is near the 200ms SLA limit, 
    # drop traffic immediately. Note: Protect node-0 in Task-3.
    #
    if worst_node.queue_depth > 120 or worst_node.latency_ms > 180:
        if not (obs.task_id == "task-3" and worst_node.node_id == "node-0"):
            return SREAction(
                action_type=ActionType.SHED_LOAD,
                target_node_id=worst_node.node_id,
                parameter=0.4  # Drop 40% of traffic
            )

    # 4. Proactive Scaling
    # Add capacity to stay ahead of the ramp (Task-1) or failure load (Task-2).
    #
    return SREAction(
        action_type=ActionType.SCALE_UP,
        target_node_id=worst_node.node_id,
        parameter=1.0  # Add 1 unit of capacity
    )

def run_baseline(task_id: str, url: str = "http://localhost:8000"):
    print(f"\n--- Running Baseline Agent on {task_id} ---")
    history = []
    
    with AntiAtroposEnv(base_url=url).sync() as env:  # ✅ FIX HERE
        result = env.reset(task_id=task_id)
        history.append(result.observation)
        
        while not result.done:
            action = decide_action(result.observation)
            result = env.step(action)
            history.append(result.observation)
            
            if result.observation.step % 20 == 0:
                print(f"Step {result.observation.step}: Queue={result.observation.total_queue_backlog}, "
                      f"Lat={result.observation.average_latency_ms:.1f}ms")

    report = grade_episode(history, task_id=task_id)
    print(f"\nFINAL GRADE: {report.summary()}")
    return report

if __name__ == "__main__":
    # Ensure your server is running before executing!
    #
    for task in ["task-1", "task-2", "task-3"]:
        try:
            run_baseline(task)
        except Exception as e:
            print(f"Could not run {task}: {e}")