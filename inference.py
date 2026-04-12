import os
import json
import re
import sys
from openai import OpenAI
from log_env import LogTriageEnvironment, Action

sys.stdout.reconfigure(line_buffering=True)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://router.huggingface.co/v1"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-72B-Instruct"
)

HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


def clean_and_parse_json(text):

    try:

        match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            return json.loads(match.group())

        numbers = re.findall(r'\d+', text)

        if numbers:
            return {"identified_log_id": int(numbers[0])}

        return {"identified_log_id": -1}

    except Exception:
        return {"identified_log_id": -1}


def run_task(difficulty):

    print(f"[START] Running task: {difficulty}")

    env = LogTriageEnvironment()

    obs = env.reset(difficulty=difficulty)

    done = False
    total_reward = 0.0

    while not done:

        prompt = (
            f"Logs: {obs.logs}\n"
            "Identify the ID of the CRITICAL log. "
            "IMPORTANT: Respond ONLY with JSON: "
            "{\"identified_log_id\": ID}"
        )

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            raw_content = response.choices[0].message.content

            action_data = clean_and_parse_json(raw_content)

            action = Action(**action_data)

            obs, reward, done, info = env.step(action)

            print(
                f"[STEP] Reward: {reward.score} "
                f"| Message: {reward.message}"
            )

            total_reward += reward.score

        except Exception as e:

            print(f"[ERROR] {e}")
            break

    print(
        f"[END] Final Score: {total_reward} "
        f"| Task: {difficulty}"
    )

    return total_reward


def run_inference(payload=None):

    results = {}

    for diff in ["easy", "medium", "hard"]:

        try:

            score = run_task(diff)

            results[diff] = score

        except Exception as e:

            results[diff] = str(e)

    return results