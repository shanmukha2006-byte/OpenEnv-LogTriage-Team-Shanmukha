import random
from dataclasses import dataclass


# Observation model
@dataclass
class Observation:
    logs: list


# Reward model
@dataclass
class Reward:
    score: float
    message: str


# Action model
@dataclass
class Action:
    identified_log_id: int


class LogTriageEnvironment:

    def __init__(self):

        self.logs = []
        self.critical_id = None
        self.done = False

    # Reset environment
    def reset(self, difficulty="easy"):

        self.done = False

        if difficulty == "easy":
            total_logs = 3

        elif difficulty == "medium":
            total_logs = 5

        else:
            total_logs = 7

        self.logs = []

        # Create logs (deterministic base)
        for i in range(total_logs):

            self.logs.append({
                "id": i,
                "severity": "INFO"
            })

        # Select one critical log
        self.critical_id = random.randint(
            0,
            total_logs - 1
        )

        self.logs[self.critical_id][
            "severity"
        ] = "CRITICAL"

        return Observation(
            logs=self.logs
        )

    # Step function
    def step(self, action):

        if self.done:

            return (
                Observation(self.logs),
                Reward(0.0, "Already done"),
                True,
                {}
            )

        self.done = True

        if action.identified_log_id == self.critical_id:

            reward = Reward(
                1.0,
                "Correct log identified"
            )

        else:

            reward = Reward(
                0.0,
                "Incorrect log identified"
            )

        return (
            Observation(self.logs),
            reward,
            True,
            {}
        )

    def state(self):

        return {
            "logs": self.logs,
            "critical_id": self.critical_id,
            "done": self.done
        }