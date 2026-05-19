# Chapter 16 Model: Landward Migration, Schooling & Selective Tidal Stream Transport
# ─────────────────────────────────────────────────────────────────────────────────
# Demonstrates integration of:
#   calculate-difficulty-landward        (Landward-Migration.nls)
#   calculate-swimming-speed-landward    (Landward-Migration.nls)
#   calculate-cost-to-home              (Landward-Migration.nls)
#   migrate-landward                    (Landward-Migration.nls)
#   school / separate / cohere / align / adjust-speed  (Schooling.nls)
#   selective-tidal-stream-transport-landward (Selective-Tidal-Stream-Transport.nls)
#   calculate-swim-energy               (Swim.nls)
#
# Schooling is nested inside migrate-landward (Case 3 of STST).
# Run: Rscript ch16_model.R

library(dplyr)
library(ggplot2)

set.seed(42)

# ── Globals ───────────────────────────────────────────────────────────────────
MAX_SEAWARD_VELOCITY  <-  1.5   # max-seaward-velocity  (difficulty normalization)
MAX_LANDWARD_VELOCITY <- -1.5   # max-landward-velocity (difficulty normalization)

# ── Environment ───────────────────────────────────────────────────────────────
# 1-D channel: patch 1 = migration-patch (downstream)
#              patch 60 = home-patch (upstream)
N_PATCHES <- 60
N_TICKS   <- 600   # 3 full 200-tick velocity cycles
N_AGENTS  <- 20

patches <- tibble(
  patch_id     = 1:N_PATCHES,
  terrain      = "water",
  velocity     = 0.0,
  depth        = runif(N_PATCHES, 0, 5),
  cost_to_home = 1e6
)

# ── Agents ────────────────────────────────────────────────────────────────────
agents <- tibble(
  id                   = 1:N_AGENTS,
  x                    = 1,
  home_patch           = N_PATCHES,
  start_migration      = TRUE,
  landward_migration   = TRUE,
  at_destination       = FALSE,
  energy               = 100.0,
  speed                = 0.1,
  prev_speed           = 0.1,
  min_speed            = 0.1,
  max_speed            = sample(100:200, N_AGENTS, replace = TRUE),
  swim_efficiency      = 1.0,
  swim_base            = 0.02,
  difficulty_factor    = 1.0,
  E_swim               = 0.0,
  weight               = sample(50:100, N_AGENTS, replace = TRUE),
  in_stst              = FALSE,
  stst_start_tick      = 0,
  heading              = 90.0,
  minimum_separation   = 0.3,
  cohere_coefficient   = 0.50,
  align_coefficient    = 0.50,
  separate_coefficient = 0.50
)


# ─────────────────────────────────────────────────────────────────────────────
# SUBMODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── Tidal velocity — 200-tick sawtooth, ±0.5 m/s ─────────────────────────────
# Positive = landward, negative = seaward. Mirrors update-tidal-velocity.
tidal_velocity <- function(tick) {
  cycle_tick <- tick %% 200
  if (cycle_tick < 100)
    0.5 - cycle_tick * (1.0 / 100)
  else
    -0.5 + (cycle_tick - 100) * (1.0 / 100)
}

# ── Least-cost routing field (mirrors calculate-cost-to-home) ─────────────────
# Cumulative difficulty-weighted cost from home outward.
# base_cost = abs(velocity) + 0.5 * max(0, depth_slope)
# mirrors compute-travel-cost.
calc_cost_to_home <- function(n_patches, difficulty, depths, velocity) {
  cost <- rep(1e6, n_patches)
  cost[n_patches] <- 0
  for (i in (n_patches - 1):1) {
    v_resistance <- abs(velocity)
    slope        <- max(0.0, depths[i + 1] - depths[i])
    base_cost    <- v_resistance + 0.5 * slope
    cost[i]      <- cost[i + 1] + base_cost * difficulty
  }
  cost
}

# ── Hydrodynamic difficulty (mirrors calculate-difficulty-landward) ───────────
calc_difficulty <- function(velocity, weight, max_weight,
                            v_max = MAX_SEAWARD_VELOCITY,
                            v_min = MAX_LANDWARD_VELOCITY,
                            k = 0.75) {
  norm_v <- (velocity - v_min) / (v_max - v_min)
  df_raw <- (norm_v / (weight / max_weight))^k
  pmin(pmax(1 + 9 * df_raw, 1), 10)
}

# ── Swimming speed (mirrors calculate-swimming-speed-landward) ────────────────
# k is negative: adverse flow reduces desired speed.
# Rate of change is capped to prevent instantaneous acceleration.
calc_swim_speed <- function(energy, difficulty, max_speed, min_speed,
                            velocity, prev_speed, swim_efficiency,
                            k = -0.75, max_change = 0.5) {
  energy_factor   <- energy / 100.0
  velocity_impact <- k * velocity * 300
  desired <- min(max_speed, max(min_speed,
              (max_speed * energy_factor / difficulty) + velocity_impact))
  max_change_eff <- max_change * swim_efficiency
  delta  <- desired - prev_speed
  speed  <- prev_speed + sign(delta) * min(abs(delta), max_change_eff)
  max(speed, min_speed)
}

# ── Swimming energy cost (mirrors calculate-swim-energy) ─────────────────────
calc_swim_energy <- function(difficulty, swim_efficiency, swim_base,
                             beta = 0.75) {
  energy_mult <- difficulty * (1 / swim_efficiency)
  swim_base * (energy_mult^beta)
}

# ── Schooling — mirrors the BOIDS block inside migrate-landward ───────────────
# Called from migrate_landward_lc only. Agents in STST never reach this.
# Returns list(heading, speed).
school_agent <- function(agent_id, agents) {
  focal <- agents[agents$id == agent_id, ]
  mates <- agents[agents$id != agent_id &
                    agents$start_migration &
                    agents$landward_migration == focal$landward_migration, ]

  if (nrow(mates) == 0)
    return(list(heading = focal$heading, speed = focal$speed))

  # find-nearest-neighbor
  dists   <- abs(mates$x - focal$x)
  nn_dist <- min(dists)
  nn_x    <- mates$x[which.min(dists)]

  new_heading <- focal$heading
  new_speed   <- focal$speed

  # separate / cohere + align
  if (nn_dist < focal$minimum_separation) {
    if (nn_x > focal$x) {
      # separate: slow down if neighbor is ahead
      if (focal$speed > focal$min_speed)
        new_speed <- focal$speed - 0.1
    } else {
      # separate: turn away from nearest neighbor
      new_heading <- focal$heading +
        (focal$separate_coefficient / 100) * sign(focal$x - nn_x) * 10
    }
  } else {
    # cohere: turn toward centroid
    centroid_x  <- mean(mates$x)
    new_heading <- focal$heading +
      (focal$cohere_coefficient / 100) * sign(centroid_x - focal$x) * 10
    # align: match mean heading
    mean_heading <- mean(mates$heading)
    new_heading  <- new_heading +
      (focal$align_coefficient / 100) * (mean_heading - new_heading)
  }

  # adjust-speed
  if (max(mates$speed) > focal$speed + 0.1) {
    new_speed <- focal$speed + focal$speed * 0.1    # speed up
  } else if (focal$speed > focal$min_speed) {
    new_speed <- focal$speed - focal$speed * 0.1    # slow down
  }
  new_speed <- max(new_speed, focal$min_speed)

  list(heading = new_heading %% 360, speed = new_speed)
}

# ── Landward migration with schooling nested inside ───────────────────────────
# Mirrors migrate-landward: BOIDS first, then move along least-cost path.
# Returns list(x, heading, speed).
migrate_landward_lc <- function(agent_id, agents) {
  school_result <- school_agent(agent_id, agents)
  focal  <- agents[agents$id == agent_id, ]
  steps  <- max(1, ceiling(school_result$speed / 3))
  new_x  <- min(focal$x + steps, focal$home_patch)
  list(x = new_x, heading = school_result$heading, speed = school_result$speed)
}

# ── STST gatekeeper (mirrors selective-tidal-stream-transport-landward) ────────
# Case 1 — cooldown active:      continue drifting, no schooling
# Case 2 — flow > swim capacity: enter STST, drift, no schooling
# Case 3 — swim capacity OK:     exit STST, call migrate_landward_lc
#                                (schooling runs inside)
# Returns list of updated agent state fields.
stst_landward <- function(agent_id, agents, velocity, current_tick) {
  focal  <- agents[agents$id == agent_id, ]
  S_swim <- focal$speed / 300.0

  # Case 1: cooldown active — continue drifting
  if (focal$in_stst && (current_tick - focal$stst_start_tick < 1)) {
    drift_dist <- abs(velocity) * (1 - focal$swim_efficiency)
    new_x      <- min(focal$x + ceiling(drift_dist), focal$home_patch)
    return(list(x = new_x, heading = focal$heading, speed = focal$speed,
                in_stst = TRUE, stst_start_tick = focal$stst_start_tick,
                energy = focal$energy))
  }

  # Case 2: flow exceeds swim capacity — enter STST
  if (abs(velocity) > S_swim && S_swim <= focal$min_speed) {
    drift_dist <- abs(velocity) * (1 - focal$swim_efficiency)
    new_x      <- min(focal$x + ceiling(drift_dist), focal$home_patch)
    return(list(x = new_x, heading = focal$heading, speed = focal$speed,
                in_stst = TRUE, stst_start_tick = current_tick,
                energy = focal$energy))
  }

  # Case 3: swim capacity sufficient — exit STST, migrate with schooling
  move   <- migrate_landward_lc(agent_id, agents)
  E_swim <- calc_swim_energy(focal$difficulty_factor,
                             focal$swim_efficiency, focal$swim_base)
  list(x = move$x, heading = move$heading, speed = move$speed,
       in_stst = FALSE, stst_start_tick = focal$stst_start_tick,
       energy = max(focal$energy - E_swim, 0))
}


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

log_ticks  <- vector("list", N_TICKS)
max_weight <- max(agents$weight)

for (tick in 1:N_TICKS) {

  # 1. Tidal velocity — sawtooth signal
  v                <- tidal_velocity(tick)
  patches$velocity <- v

  # 2. Rebuild least-cost field from home outward
  mean_diff            <- calc_difficulty(v, mean(agents$weight), max_weight)
  patches$cost_to_home <- calc_cost_to_home(N_PATCHES, mean_diff,
                                            patches$depth, v)

  # 3. Per-agent difficulty and swimming speed (vectorised)
  agents <- agents |>
    mutate(
      difficulty_factor = calc_difficulty(v, weight, max_weight),
      speed = mapply(calc_swim_speed, energy, difficulty_factor,
                     max_speed, min_speed, prev_speed, swim_efficiency,
                     MoreArgs = list(velocity = v)),
      prev_speed = speed
    )

  # 4. STST gatekeeper (calls migrate_landward_lc with schooling for Case 3)
  for (i in seq_len(nrow(agents))) {
    if (!agents$start_migration[i] || !agents$landward_migration[i] ||
        agents$at_destination[i]) next

    result <- stst_landward(
      agent_id     = agents$id[i],
      agents       = agents,
      velocity     = v,
      current_tick = tick
    )

    agents$x[i]               <- result$x
    agents$heading[i]         <- result$heading
    agents$speed[i]           <- result$speed
    agents$in_stst[i]         <- result$in_stst
    agents$stst_start_tick[i] <- result$stst_start_tick
    agents$energy[i]          <- result$energy
  }

  # 5. Baseline swimming energy cost (all agents, including drift ticks)
  agents <- agents |>
    mutate(
      E_swim = calc_swim_energy(difficulty_factor, swim_efficiency, swim_base),
      energy = pmax(energy - E_swim, 0)
    )

  # 6. Arrival check
  agents <- agents |>
    mutate(
      at_destination     = x >= home_patch,
      landward_migration = !at_destination
    )

  log_ticks[[tick]] <- tibble(
    tick        = tick,
    velocity    = v,
    mean_x      = mean(agents$x),
    mean_energy = mean(agents$energy),
    n_stst      = sum(agents$in_stst),
    n_arrived   = sum(agents$at_destination)
  )

  # Early stop once all agents have arrived
  if (all(agents$at_destination)) {
    message("All agents arrived at tick ", tick)
    log_ticks <- log_ticks[1:tick]
    break
  }
}

results <- bind_rows(log_ticks)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

p1 <- ggplot(results, aes(x = tick)) +
  geom_line(aes(y = mean_x, color = "Mean position"), linewidth = 1) +
  geom_line(aes(y = velocity * 20 + 30, color = "Tidal velocity (scaled)"),
            linetype = "dashed", linewidth = 0.6) +
  scale_color_manual(values = c("Mean position"           = "#2c7bb6",
                                "Tidal velocity (scaled)" = "#d7191c")) +
  labs(title = "Migration progress", x = "Tick",
       y = "Mean upstream position (patch index)", color = NULL) +
  theme_minimal()

p2 <- ggplot(results, aes(x = tick, y = n_stst)) +
  geom_area(fill = "#fdae61", alpha = 0.6) +
  geom_line(color = "#f46d43", linewidth = 0.8) +
  labs(title = "Agents in STST per tick", x = "Tick",
       y = "Count in passive drift") +
  theme_minimal()

p3 <- ggplot(results, aes(x = tick, y = mean_energy)) +
  geom_line(color = "#1a9641", linewidth = 1) +
  labs(title = "Mean agent energy", x = "Tick", y = "Energy") +
  theme_minimal()

# Combine and save
library(patchwork)
combined <- p1 + p2 + p3 +
  plot_annotation(title = "Chapter 16: Landward Migration + Schooling + STST")

ggsave("E:/Sample_ABM_Tutorials/Complex_Example_02/R/ch16_output.png", combined, width = 14, height = 4, dpi = 150)
message("Plot saved to ch16_output.png")

# Summary
message("\n── Summary ──────────────────────────────────────────")
message("Ticks run:            ", nrow(results))
message("Agents arrived:       ", sum(agents$at_destination), " / ", N_AGENTS)
message("Mean final energy:    ", round(mean(agents$energy), 2))
message("Max STST in one tick: ", max(results$n_stst))
