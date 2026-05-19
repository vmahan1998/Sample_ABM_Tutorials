"""
Chapter 16 Model: Landward Migration, Schooling & Selective Tidal Stream Transport
------------------------------------------------------------------------------------
A 1-D abstract channel model demonstrating the integration of:
  - calculate-difficulty-landward  (Landward-Migration.nls)
  - calculate-swimming-speed-landward (Landward-Migration.nls)
  - calculate-cost-to-home         (Landward-Migration.nls)
  - migrate-landward               (Landward-Migration.nls)
  - school / find-schoolmates / separate / cohere / align / adjust-speed
                                   (Schooling.nls and component files)
  - selective-tidal-stream-transport-landward (Selective-Tidal-Stream-Transport.nls)
  - calculate-swim-energy          (Swim.nls)

Schooling is nested inside migrate-landward (Case 3 of STST).
Run:  python ch16_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Globals ───────────────────────────────────────────────────────────────────
MAX_SEAWARD_VELOCITY  =  1.5   # max-seaward-velocity  (used by difficulty calc)
MAX_LANDWARD_VELOCITY = -1.5   # max-landward-velocity (used by difficulty calc)

rng = np.random.default_rng(seed=42)

# ── Environment ───────────────────────────────────────────────────────────────
# 1-D channel: index 0 = migration-patch (downstream)
#              index 59 = home-patch (upstream)
N_PATCHES = 60
N_TICKS   = 600   # 3 full 200-tick velocity cycles
N_AGENTS  = 20

patches = pd.DataFrame({
    "patch_id"    : np.arange(N_PATCHES),
    "terrain"     : "water",
    "velocity"    : 0.0,
    "depth"       : rng.uniform(0, 5, N_PATCHES),
    "cost_to_home": 1e6,
})

# ── Agents ────────────────────────────────────────────────────────────────────
agents = pd.DataFrame({
    "id"                  : np.arange(N_AGENTS),
    "x"                   : 0,
    "home_patch"          : N_PATCHES - 1,
    "start_migration"     : True,
    "landward_migration"  : True,
    "at_destination"      : False,
    "energy"              : 100.0,
    "speed"               : 0.1,
    "prev_speed"          : 0.1,
    "min_speed"           : 0.1,
    "max_speed"           : rng.integers(100, 200, N_AGENTS).astype(float),
    "swim_efficiency"     : 1.0,
    "swim_base"           : 0.02,
    "difficulty_factor"   : 1.0,
    "E_swim"              : 0.0,
    "weight"              : rng.integers(50, 100, N_AGENTS).astype(float),
    "in_stst"             : False,
    "stst_start_tick"     : 0,
    "heading"             : 90.0,
    "minimum_separation"  : 0.3,
    "cohere_coefficient"  : 0.50,
    "align_coefficient"   : 0.50,
    "separate_coefficient": 0.50,
})


# ─────────────────────────────────────────────────────────────────────────────
# SUBMODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def tidal_velocity(tick):
    """
    200-tick sawtooth, ±0.5 m/s.
    Positive = landward flow, negative = seaward flow.
    Mirrors update-tidal-velocity.
    """
    cycle_tick = tick % 200
    if cycle_tick < 100:
        return 0.5 - cycle_tick * (1.0 / 100)
    else:
        return -0.5 + (cycle_tick - 100) * (1.0 / 100)


def calc_cost_to_home(n_patches, difficulty, depths, velocity):
    """
    Cumulative difficulty-weighted cost field from home outward.
    Mirrors calculate-cost-to-home / compute-travel-cost:
      base_cost = abs(velocity) + 0.5 * max(0, depth_slope)
    """
    cost = np.full(n_patches, 1e6)
    cost[n_patches - 1] = 0.0
    for i in range(n_patches - 2, -1, -1):
        v_resistance = abs(velocity)
        slope        = max(0.0, depths[i + 1] - depths[i])
        base_cost    = v_resistance + 0.5 * slope
        cost[i]      = cost[i + 1] + base_cost * difficulty
    return cost


def calc_difficulty(velocity, weight, max_weight,
                    v_max=MAX_SEAWARD_VELOCITY,
                    v_min=MAX_LANDWARD_VELOCITY,
                    k=0.75):
    """
    Mirrors calculate-difficulty-landward.
    Normalizes velocity against observed range, scales by agent weight.
    """
    norm_v = (velocity - v_min) / (v_max - v_min)
    df_raw = (norm_v / (weight / max_weight)) ** k
    return np.clip(1 + 9 * df_raw, 1, 10)


def calc_swim_speed(energy, difficulty, max_speed, min_speed,
                    velocity, prev_speed, swim_efficiency,
                    k=-0.75, max_change=0.5):
    """
    Mirrors calculate-swimming-speed-landward.
    k is negative: adverse flow reduces desired speed.
    Rate of change is capped to prevent instantaneous acceleration.
    """
    energy_factor   = energy / 100.0
    velocity_impact = k * velocity * 300
    desired = float(np.clip(
        (max_speed * energy_factor / difficulty) + velocity_impact,
        min_speed, max_speed
    ))
    max_change_eff = max_change * swim_efficiency
    delta = desired - prev_speed
    speed = prev_speed + np.sign(delta) * min(abs(delta), max_change_eff)
    return max(float(speed), min_speed)


def calc_swim_energy(difficulty, swim_efficiency, swim_base, beta=0.75):
    """
    Mirrors calculate-swim-energy.
    Higher difficulty and lower efficiency increase energy cost.
    """
    energy_mult = difficulty * (1.0 / swim_efficiency)
    return swim_base * (energy_mult ** beta)


def school_agent(agent_id, agents):
    """
    Mirrors the BOIDS block inside migrate-landward:
      find-schoolmates → find-nearest-neighbor →
      separate / (cohere + align) → adjust-speed

    Called from migrate_landward_lc only.
    Agents in STST never reach this function.
    Returns (new_heading, new_speed).
    """
    focal = agents.loc[agents["id"] == agent_id].iloc[0]
    mates = agents[
        (agents["id"] != agent_id) &
        (agents["start_migration"]) &
        (agents["landward_migration"] == focal["landward_migration"])
    ]

    if mates.empty:
        return focal["heading"], focal["speed"]

    # find-nearest-neighbor
    dists   = np.abs(mates["x"] - focal["x"])
    nn_dist = dists.min()
    nn_x    = mates.loc[dists.idxmin(), "x"]

    new_heading = focal["heading"]
    new_speed   = focal["speed"]

    # separate / cohere + align
    if nn_dist < focal["minimum_separation"]:
        if nn_x > focal["x"]:
            # separate: slow down if neighbor ahead (in-cone check)
            if focal["speed"] > focal["min_speed"]:
                new_speed = focal["speed"] - 0.1
        else:
            # separate: turn away from nearest neighbor
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

    # adjust-speed
    if mates["speed"].max() > focal["speed"] + 0.1:
        new_speed = focal["speed"] + focal["speed"] * 0.1   # speed up
    elif focal["speed"] > focal["min_speed"]:
        new_speed = focal["speed"] - focal["speed"] * 0.1   # slow down
    new_speed = max(new_speed, focal["min_speed"])

    return new_heading % 360, new_speed


def migrate_landward_lc(agent_id, agents):
    """
    Mirrors migrate-landward: schooling (BOIDS) first, then path-following.
    Steps upstream by ceil(speed / 3) patches along least-cost path.
    Returns (new_x, new_heading, new_speed).
    """
    new_heading, new_speed = school_agent(agent_id, agents)
    focal  = agents.loc[agents["id"] == agent_id].iloc[0]
    steps  = max(1, int(np.ceil(new_speed / 3)))
    new_x  = min(int(focal["x"]) + steps, int(focal["home_patch"]))
    return new_x, new_heading, new_speed


def stst_landward(agent_id, agents, velocity, current_tick):
    """
    Mirrors selective-tidal-stream-transport-landward.

    Case 1 — cooldown active:      continue drifting, no schooling
    Case 2 — flow > swim capacity: enter STST, drift, no schooling
    Case 3 — swim capacity OK:     exit STST, call migrate_landward_lc
                                   (schooling runs inside)

    Returns a dict of updated agent state fields.
    """
    focal  = agents.loc[agents["id"] == agent_id].iloc[0]
    S_swim = focal["speed"] / 300.0   # m/s effective swim speed

    # Case 1: still in STST cooldown — continue drifting
    if focal["in_stst"] and (current_tick - focal["stst_start_tick"] < 1):
        drift_dist = abs(velocity) * (1 - focal["swim_efficiency"])
        new_x      = min(int(focal["x"]) + int(np.ceil(drift_dist)),
                         int(focal["home_patch"]))
        return {
            "x": new_x, "heading": focal["heading"], "speed": focal["speed"],
            "in_stst": True, "stst_start_tick": focal["stst_start_tick"],
            "energy": focal["energy"],
        }

    # Case 2: flow exceeds swim capacity — enter STST
    if abs(velocity) > S_swim and S_swim <= focal["min_speed"]:
        drift_dist = abs(velocity) * (1 - focal["swim_efficiency"])
        new_x      = min(int(focal["x"]) + int(np.ceil(drift_dist)),
                         int(focal["home_patch"]))
        return {
            "x": new_x, "heading": focal["heading"], "speed": focal["speed"],
            "in_stst": True, "stst_start_tick": current_tick,
            "energy": focal["energy"],
        }

    # Case 3: swim capacity sufficient — exit STST, migrate with schooling
    new_x, new_heading, new_speed = migrate_landward_lc(agent_id, agents)
    E_swim     = calc_swim_energy(focal["difficulty_factor"],
                                  focal["swim_efficiency"], focal["swim_base"])
    new_energy = max(focal["energy"] - E_swim, 0.0)
    return {
        "x": new_x, "heading": new_heading, "speed": new_speed,
        "in_stst": False, "stst_start_tick": focal["stst_start_tick"],
        "energy": new_energy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

log_ticks  = []
max_weight = agents["weight"].max()

for tick in range(1, N_TICKS + 1):

    # 1. Tidal velocity — sawtooth signal
    v                   = tidal_velocity(tick)
    patches["velocity"] = v

    # 2. Rebuild least-cost field from home outward
    mean_diff               = float(calc_difficulty(v, agents["weight"].mean(),
                                                    max_weight))
    patches["cost_to_home"] = calc_cost_to_home(N_PATCHES, mean_diff,
                                                patches["depth"].values, v)

    # 3. Per-agent difficulty and swimming speed
    agents["difficulty_factor"] = calc_difficulty(
        v, agents["weight"].values, max_weight
    )
    agents["speed"] = [
        calc_swim_speed(
            agents.at[i, "energy"],
            agents.at[i, "difficulty_factor"],
            agents.at[i, "max_speed"],
            agents.at[i, "min_speed"],
            v,
            agents.at[i, "prev_speed"],
            agents.at[i, "swim_efficiency"],
        )
        for i in agents.index
    ]
    agents["prev_speed"] = agents["speed"]

    # 4. STST gatekeeper (calls migrate_landward_lc with schooling for Case 3)
    active = (
        agents["start_migration"] &
        agents["landward_migration"] &
        ~agents["at_destination"]
    )
    for i in agents.index[active]:
        result = stst_landward(
            agent_id     = agents.at[i, "id"],
            agents       = agents,
            velocity     = v,
            current_tick = tick,
        )
        agents.at[i, "x"]               = result["x"]
        agents.at[i, "heading"]         = result["heading"]
        agents.at[i, "speed"]           = result["speed"]
        agents.at[i, "in_stst"]         = result["in_stst"]
        agents.at[i, "stst_start_tick"] = result["stst_start_tick"]
        agents.at[i, "energy"]          = result["energy"]

    # 5. Baseline swimming energy cost (all agents, including drift ticks)
    agents["E_swim"] = [
        calc_swim_energy(
            agents.at[i, "difficulty_factor"],
            agents.at[i, "swim_efficiency"],
            agents.at[i, "swim_base"],
        )
        for i in agents.index
    ]
    agents["energy"] = np.maximum(agents["energy"] - agents["E_swim"], 0)

    # 6. Arrival check
    agents["at_destination"]     = agents["x"] >= agents["home_patch"]
    agents["landward_migration"] = ~agents["at_destination"]

    log_ticks.append({
        "tick"       : tick,
        "velocity"   : v,
        "mean_x"     : agents["x"].mean(),
        "mean_energy": agents["energy"].mean(),
        "n_stst"     : int(agents["in_stst"].sum()),
        "n_arrived"  : int(agents["at_destination"].sum()),
    })

    # Early stop once all agents have arrived
    if agents["at_destination"].all():
        print(f"All agents arrived at tick {tick}")
        break

results = pd.DataFrame(log_ticks)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Chapter 16: Landward Migration + Schooling + STST", fontsize=11)

# Migration progress with tidal velocity overlay
ax = axes[0]
ax.plot(results["tick"], results["mean_x"],
        color="#2c7bb6", linewidth=1.5, label="Mean position")
ax.plot(results["tick"], results["velocity"] * 20 + 30,
        color="#d7191c", linewidth=0.8, linestyle="--",
        label="Tidal velocity (scaled)")
ax.set_title("Migration progress")
ax.set_xlabel("Tick")
ax.set_ylabel("Mean upstream position (patch index)")
ax.legend(fontsize=8)

# Agents in STST per tick
ax = axes[1]
ax.fill_between(results["tick"], results["n_stst"],
                color="#fdae61", alpha=0.6)
ax.plot(results["tick"], results["n_stst"],
        color="#f46d43", linewidth=0.8)
ax.set_title("Agents in STST per tick")
ax.set_xlabel("Tick")
ax.set_ylabel("Count in passive drift")

# Mean energy over time
ax = axes[2]
ax.plot(results["tick"], results["mean_energy"],
        color="#1a9641", linewidth=1.5)
ax.set_title("Mean agent energy")
ax.set_xlabel("Tick")
ax.set_ylabel("Energy")

plt.tight_layout()
plt.savefig("ch16_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to ch16_output.png")

# Quick summary
print("\n── Summary ──────────────────────────────────────────")
print(f"Ticks run:         {len(results)}")
print(f"Agents arrived:    {int(agents['at_destination'].sum())} / {N_AGENTS}")
print(f"Mean final energy: {agents['energy'].mean():.2f}")
print(f"Max STST in one tick: {results['n_stst'].max()}")
