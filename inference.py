from log_env import LogTriageEnvironment, Action


def run_task(difficulty):

    print(f"[START] task={difficulty}", flush=True)

    env = LogTriageEnvironment()

    obs = env.reset(difficulty=difficulty)

    done = False
    total_reward = 0.0
    step_count = 0

    while not done:

        critical_id = -1

        for log in obs.logs:
            if log["severity"] == "CRITICAL":
                critical_id = log["id"]
                break

        action = Action(
            identified_log_id=critical_id
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
            score = run_task(diff)
            results[diff] = score

        except Exception as e:
            results[diff] = str(e)

    return results
