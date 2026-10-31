"""Multi-armed bandit for budget nudges: a reinforcement-learning view of 'which tip works best?'."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUDGES = [
    dict(id="food_cap", text="Cap food-delivery orders at a monthly limit", category="Food Delivery",
         accept=0.55, saving=0.20),
    dict(id="cook_home", text="Cook at home two extra days a week", category="Dining", accept=0.35, saving=0.25),
    dict(id="sub_audit", text="Cancel subscriptions you rarely use", category="Subscriptions",
         accept=0.60, saving=0.30),
    dict(id="wait_rule", text="Apply a 24-hour wait rule before online purchases", category="Shopping",
         accept=0.45, saving=0.20),
    dict(id="metro", text="Replace short cab rides with metro or walking", category="Transport",
         accept=0.50, saving=0.25),
    dict(id="grocery_list", text="Plan a weekly grocery list", category="Groceries", accept=0.65, saving=0.10),
    dict(id="carpool", text="Carpool or combine fuel trips", category="Fuel", accept=0.40, saving=0.15),
]
POLICIES = ["Random", "Epsilon-greedy", "UCB1", "Thompson sampling"]


def nudge_table(monthly_spend: pd.Series) -> pd.DataFrame:
    rows = []
    for n in NUDGES:
        spend = float(monthly_spend.get(n["category"], 0.0))
        if spend <= 0:
            continue
        potential = spend * n["saving"]
        rows.append({**n, "monthly_spend": spend, "potential_saving": potential,
                     "expected_saving": potential * n["accept"]})
    table = pd.DataFrame(rows)
    table["success_prob"] = table["accept"] * table["potential_saving"] / table["potential_saving"].max()
    return table.sort_values("expected_saving", ascending=False).reset_index(drop=True)


def _choose(policy: str, t: int, counts, sums, alpha, beta, rng, epsilon: float) -> int:
    k = len(counts)
    if policy == "Random":
        return int(rng.integers(k))
    if t < k:
        return t
    if policy == "Epsilon-greedy":
        return int(rng.integers(k)) if rng.random() < epsilon else int(np.argmax(sums / counts))
    if policy == "UCB1":
        return int(np.argmax(sums / counts + np.sqrt(2 * np.log(t + 1) / counts)))
    return int(np.argmax(rng.beta(alpha, beta)))


def simulate(p: np.ndarray, policy: str, rounds: int = 400, seed: int = 0, epsilon: float = 0.1) -> dict:
    rng = np.random.default_rng(seed)
    k = len(p)
    counts, sums = np.zeros(k), np.zeros(k)
    alpha, beta = np.ones(k), np.ones(k)
    actions, rewards = np.empty(rounds, int), np.empty(rounds)
    for t in range(rounds):
        a = _choose(policy, t, np.maximum(counts, 1e-9), sums, alpha, beta, rng, epsilon)
        r = float(rng.random() < p[a])
        counts[a] += 1
        sums[a] += r
        alpha[a] += r
        beta[a] += 1 - r
        actions[t], rewards[t] = a, r
    return {"actions": actions, "rewards": rewards, "estimates": sums / np.maximum(counts, 1), "counts": counts}


def run_experiment(p: np.ndarray, rounds: int = 400, runs: int = 40, epsilon: float = 0.1) -> dict:
    best = p.max()
    regret = {}
    pulls = {}
    for policy in POLICIES:
        curves, counts = [], np.zeros(len(p))
        for seed in range(runs):
            result = simulate(p, policy, rounds, seed, epsilon)
            curves.append(np.cumsum(best - p[result["actions"]]))
            counts += result["counts"]
        regret[policy] = np.mean(curves, axis=0)
        pulls[policy] = counts / counts.sum()
    return {"regret": pd.DataFrame(regret), "pulls": pd.DataFrame(pulls)}
