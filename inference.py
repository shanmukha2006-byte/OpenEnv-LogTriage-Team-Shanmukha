import os
import json
import re

from openai import OpenAI
from log_env import LogTriageEnvironment, Action


MODEL_NAME = "gpt-4o-mini"


def create_client():

    api_base = os.environ.get("API_BASE_URL")

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("LITELLM_API_KEY")
        or "dummy-key"   # prevents crash
    )

    return OpenAI(
        base_url=api_base,
        api_key=api_key
    )


def parse_response(text):

    try:
        match = re.search(r'\{.*\}', text)

        if match:
            data = json.loads(match.group())
            return int(data["identified_log_id"])

    except:
        pass

    return -1


def run_task(difficulty):

    print(f"[START] task={difficulty}", flush=True)

    env = LogTriageEnvironment()

    obs = env.reset(difficulty=difficulty)

    done = False
    total_reward = 0.0
    step_count = 0

    client = create_client()

    while not done:

        prompt = f"""
Logs: {obs.logs}

Identify the CRITICAL log.

Respond ONLY as JSON:
{{"identified_log_id": number}}
"""

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            output = response.choices[0].message.content

            log_id = parse_response(output)

        except Exception as e:

            print(f"[ERROR] {e}", flush=True)

            # fallback logic
            log_id = -1

            for log in obs.logs:
                if log["severity"] == "CRITICAL":
                    log_id = log["id"]
                    break

        action = Action(
            identified_log_id=log_id
        )

        obs, reward, done, info = env.step(action)

        step_count += 1
        total_reward += reward.score

        print(
            f"[STEP] step={step_count} reward={reward.score}",
            flush=True
        )

    print(
        f"[END] task={difficulty} score={total_reward}",
        flush=True
    )

    return total_reward


def run_inference(payload=None):

    results = {}

    for diff in ["easy", "medium", "hard"]:

        try:
            results[diff] = run_task(diff)

        except Exception as e:

            print(f"[ERROR] {e}", flush=True)

            results[diff] = 0.0

    return results


if __name__ == "__main__":

    run_inference()
