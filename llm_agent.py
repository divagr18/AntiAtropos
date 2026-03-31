import os
import json
import textwrap
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from AntiAtropos.client import AntiAtroposEnv
from AntiAtropos.models import SREAction, ActionType
from AntiAtropos.grader import EpisodeGrader

load_dotenv()

# We satisfy the hackathon's "Must use OpenAI client" rule, 
# but proxy it directly to the Groq LPU engine for insane speed.
API_BASE_URL = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant" 

MAX_STEPS = 100
REMOTE_SPACE_URL = "https://pranavkk-antiatropos.hf.space"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous Site Reliability Engineer (SRE) agent controlling a live microservice cluster.
    You will receive a JSON state containing normalized [0.0, 1.0] queue depths (per-node and total), latencies, and traffic rates.
    
    You must intelligently balance the Lyapunov Energy (stability) against infrastructure cost and SLAs.
    Note: Booting infrastructure (SCALE_UP) takes exactly 5 ticks. Act proactively.
    
    Reply ONLY with a strictly formatted JSON object matching this schema. No markdown, no explanations, no text before or after.
    {
      "action_type": "SCALE_UP" | "SCALE_DOWN" | "REROUTE_TRAFFIC" | "SHED_LOAD" | "NO_OP",
      "target_node_id": "node-1",
      "parameter": 1.0
    }
    """
).strip()

def extract_json_action(response_text: str) -> SREAction:
    """Parses chaotic LLM output into a rigid strict Pydantic model"""
    try:
        text = response_text.replace("```json", "").replace("```", "").strip()
        payload = json.loads(text)
        if payload.get("target_node_id") is None:
            payload["target_node_id"] = "node-0"
        if payload.get("parameter") is None:
            payload["parameter"] = 0.0
        return SREAction(**payload)
    except Exception as e:
        print(f"  [!] Failed to parse LLM JSON ({e}). Text was: {response_text[:50]}... Falling back to NO_OP.")
        return SREAction(action_type=ActionType.NO_OP, target_node_id="node-0", parameter=0.0)

async def loop_episode(client: AsyncOpenAI, env: AntiAtroposEnv, task_id: str):
    print(f"\n--- Running Zero-Shot Gemini ({MODEL_NAME}) on {task_id} ---")
    
    # 1. Start the episode
    result = await env.reset(task_id=task_id)
    obs = result.observation
    
    grader = EpisodeGrader(task_id=task_id)
    grader.record(obs)
    
    history = []
    
    for step in range(MAX_STEPS):
        if result.done:
            break
            
        # 2. Get LLM Action
        state_json = obs.model_dump_json(indent=2)
        history_context = "\n".join(history[-3:]) if history else "None"
        
        user_prompt = f"Step: {step}\nLast 3 Actions:\n{history_context}\n\nCurrent Cluster State:\n{state_json}\nDecide the next action."
        
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"API Error. {e}")
            response_text = '{"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0}'
            
        action = extract_json_action(response_text)
        
        # 3. Apply action
        result = await env.step(action)
        obs = result.observation
        grader.record(obs)
        
        reward = result.reward
        log_line = f"Step {step}: Gemini chose {action.action_type.value} on {action.target_node_id} (param: {action.parameter}) -> Reward: {reward:.4f}"
        history.append(log_line)
        
        if step > 0 and step % 20 == 0:
            print(f"Step {step}: Total Backlog={obs.total_queue_backlog:.3f}, Latency={obs.average_latency_ms:.3f} (norm) | Action: {action.action_type.value}")
            
    # 4. Final Scoring
    grade = grader.score()
    print(f"\nFINAL GRADE: {grade.summary()}")

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: GROQ_API_KEY environment variable not found in .env!")
        exit(1)
        
    # We use the official OpenAI syntax, routed straight to Groq!
    llm_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    sre_env = AntiAtroposEnv(REMOTE_SPACE_URL, message_timeout_s=300)
    
    # We'll run it on all three tasks
    for task in ["task-1", "task-2", "task-3"]:
        asyncio.run(loop_episode(llm_client, sre_env, task))
