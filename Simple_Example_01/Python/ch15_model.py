"""
Chapter 15 Model: Schooling & Landward Migration
─────────────────────────────────────────────────
Demonstrates the simplest function integration:
  - migrate-landward  (face home-patch, fd speed — no velocity, cost field, or energy)
  - find-schoolmates  (Find-Schoolmates.nls)
  - find-nearest-neighbor (Find-nearest-neighbor.nls)
  - separate / cohere / align / adjust-speed (BOIDS, Schooling.nls component files)
  - school            (Schooling.nls — calls the four above)

Schooling runs AFTER migration each tick, adjusting heading and speed
for the next time step. No tidal signal, no routing, no energy.

Run:  python ch15_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Initialisation ────────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

N_AGENTS = 20
N_TICKS  = 150

agents = pd.DataFrame({
    "id"                  : np.arange(N_AGENTS),
    "x"                   : rng.uniform(0, 5, N_AGENTS),    # random start positions
    "y"                   : rng.uniform(-5, 5, N_AGENTS),
    "home_x"              : 50.0,                           # upstream destination
    "start_migration"     : True,
    "landward_migration"  : True,
    "at_destination"      : False,
    "speed"               : rng.uniform(1, 2, N_AGENTS),
    "min_speed"           : 0.5,
    "heading"             : 0.0,                            # degrees; 0 = upstream
    "minimum_separation"  : 1.5,
    "cohere_coefficient"  : 0.25,
    "align_coefficient"   : 0.25,
    "separate_coefficient": 0.25,
})


# ─────────────────────────────────────────────────────────────────────────────
# SUBMODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def migrate_landward(x, y, speed, home_x):
    """
    Face home_x and step forward by speed.
    Mirrors: face home-patch / fd speed
    No velocity, cost field, or energy — just direction and distance.
    """
    angle   = np.arctan2(0, home_x - x)      # heading toward home (y held at 0)
    new_x   = x + speed * np.cos(angle)
    new_x   = min(new_x, home_x)             # cap at destination
    heading = np.degrees(angle)
    return new_x, y, heading


def find_schoolmates(agent_id, agents):
    """
    Returns all agents sharing the same migration state as the focal agent.
    Mirrors: find-schoolmates (Find-Schoolmates.nls)
    """
    focal = agents.loc[agents["id"] == agent_id].iloc[0]
    return agents[
        (agents["id"] != agent_id) &
        (agents["start_migration"]) &
        (agents["landward_migration"] == focal["landward_migration"])
    ]


def school_agent(agent_id, agents):
    """
    Applies BOIDS rules in order:
      find-nearest-neighbor → separate / (cohere + align) → adjust-speed
    Mirrors: school (Schooling.nls) and its component files.
    Returns (new_heading, new_speed).
    """
    focal = agents.loc[agents["id"] == agent_id].iloc[0]
    mates = find_schoolmates(agent_id, agents)

    if mates.empty:
        return focal["heading"], focal["speed"]

    # find-nearest-neighbor
    dists   = np.sqrt((mates["x"] - focal["x"])**2 +
                      (mates["y"] - focal["y"])**2)
    nn_dist = dists.min()
    nn_x    = mates.loc[dists.idxmin(), "x"]

    new_heading = focal["heading"]
    new_speed   = focal["speed"]

    if nn_dist < focal["minimum_separation"]:
        # separate: slow down if neighbor is ahead, turn away otherwise
        if nn_x > focal["x"]:
            new_speed = max(focal["min_speed"], focal["speed"] - 0.1)
        else:
            new_heading = focal["heading"] + (
                focal["separate_coefficient"] / 100
            ) * np.sign(focal["x"] - nn_x) * 10
    else:
        # cohere: turn toward centroid of schoolmates
        centroid_x  = mates["x"].mean()
        new_heading = focal["heading"] + (
            focal["cohere_coefficient"] / 100
        ) * np.sign(centroid_x - focal["x"]) * 10

        # align: match mean heading of schoolmates
        mean_heading = mates["heading"].mean()
        new_heading  = new_heading + (
            focal["align_coefficient"] / 100
        ) * (mean_heading - new_heading)

    # adjust-speed: speed up if a schoolmate is faster, otherwise slow down
    if mates["speed"].max() > focal["speed"] + 0.1:
        new_speed = focal["speed"] * 1.1
    elif focal["speed"] > focal["min_speed"]:
        new_speed = focal["speed"] * 0.9
    new_speed = max(new_speed, focal["min_speed"])

    return new_heading % 360, new_speed


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

log_ticks = []

for tick in range(1, N_TICKS + 1):

    # Step 1: migrate — face home and step forward
    migrating = (
        agents["start_migration"] &
        agents["landward_migration"] &
        ~agents["at_destination"]
    )

    for i in agents.index[migrating]:
        row = agents.loc[i]
        new_x, new_y, new_heading = migrate_landward(
            row["x"], row["y"], row["speed"], row["home_x"]
        )
        agents.at[i, "x"]       = new_x
        agents.at[i, "y"]       = new_y
        agents.at[i, "heading"] = new_heading

    # Step 2: school — separation, cohesion, alignment, speed adjustment
    # Applied to all agents so schooling can slow arrived agents too;
    # agents with no schoolmates return unchanged (mates.empty branch).
    school_results    = [school_agent(aid, agents) for aid in agents["id"]]
    agents["heading"] = [r[0] for r in school_results]
    agents["speed"]   = [r[1] for r in school_results]

    # Step 3: check arrival
    agents["at_destination"]     = agents["x"] >= agents["home_x"]
    agents["landward_migration"]  = ~agents["at_destination"]

    log_ticks.append({
        "tick"      : tick,
        "mean_x"    : agents["x"].mean(),
        "mean_speed": agents["speed"].mean(),
        "n_arrived" : int(agents["at_destination"].sum()),
    })

    # Early stop once all agents have arrived
    if agents["at_destination"].all():
        print(f"All agents arrived at tick {tick}")
        break

results = pd.DataFrame(log_ticks)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Chapter 15: Schooling & Landward Migration", fontsize=11)

# Migration progress
axes[0].plot(results["tick"], results["mean_x"],
             color="#2c7bb6", linewidth=1.5)
axes[0].set_title("Mean agent position over time")
axes[0].set_xlabel("Tick")
axes[0].set_ylabel("Mean upstream position")

# Speed convergence from schooling
axes[1].plot(results["tick"], results["mean_speed"],
             color="#d7191c", linewidth=1.5)
axes[1].set_title("Mean agent speed over time")
axes[1].set_xlabel("Tick")
axes[1].set_ylabel("Mean speed")

# Cumulative arrivals
axes[2].step(results["tick"], results["n_arrived"],
             color="#1a9641", linewidth=1.5, where="post")
axes[2].set_title("Cumulative arrivals at home patch")
axes[2].set_xlabel("Tick")
axes[2].set_ylabel("Number of agents arrived")

plt.tight_layout()
plt.savefig("ch15_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to ch15_output.png")

# Summary
print("\n── Summary ──────────────────────────────────────────")
print(f"Ticks run:         {len(results)}")
print(f"Agents arrived:    {int(agents['at_destination'].sum())} / {N_AGENTS}")
print(f"Mean final speed:  {agents['speed'].mean():.3f}")
