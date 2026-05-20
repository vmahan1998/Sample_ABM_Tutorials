# Chapter 15 Model: Schooling & Landward Migration
# ─────────────────────────────────────────────────
# Demonstrates the simplest function integration:
#   migrate-landward      (face home-patch, fd speed — no velocity, cost, or energy)
#   find-schoolmates      (Find-Schoolmates.nls)
#   find-nearest-neighbor (Find-nearest-neighbor.nls)
#   separate / cohere / align / adjust-speed (Schooling.nls component files)
#   school                (Schooling.nls — calls the four above)
#
# Schooling runs AFTER migration each tick, adjusting heading and speed
# for the next time step. No tidal signal, no routing, no energy.
#
# Run: Rscript ch15_model.R

library(dplyr)
library(ggplot2)

set.seed(42)

# ── Initialisation ─────────────────────────────────────────────────────────────
N_AGENTS <- 20
N_TICKS  <- 150

agents <- tibble(
  id                   = 1:N_AGENTS,
  x                    = runif(N_AGENTS, 0, 5),    # random start positions
  y                    = runif(N_AGENTS, -5, 5),
  home_x               = 50,                       # upstream destination
  start_migration      = TRUE,
  landward_migration   = TRUE,
  at_destination       = FALSE,
  speed                = runif(N_AGENTS, 1, 2),
  min_speed            = 0.5,
  heading              = 0,                        # degrees; 0 = upstream
  minimum_separation   = 1.5,
  cohere_coefficient   = 0.25,
  align_coefficient    = 0.25,
  separate_coefficient = 0.25
)


# ─────────────────────────────────────────────────────────────────────────────
# SUBMODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── Landward migration ────────────────────────────────────────────────────────
# Face home_x and step forward by speed.
# Mirrors: face home-patch / fd speed
# No velocity, cost field, or energy — just direction and distance.
migrate_landward <- function(x, y, speed, home_x) {
  angle <- atan2(0, home_x - x)        # heading toward home (y held at 0)
  new_x <- x + speed * cos(angle)
  new_x <- pmin(new_x, home_x)         # cap at destination
  list(x = new_x, y = y, heading = angle * 180 / pi)
}

# ── Find schoolmates (mirrors Find-Schoolmates.nls) ───────────────────────────
# Returns all agents sharing the same migration state as the focal agent.
find_schoolmates <- function(agent_id, agents) {
  focal <- agents[agents$id == agent_id, ]
  agents[agents$id != agent_id &
           agents$start_migration &
           agents$landward_migration == focal$landward_migration, ]
}

# ── School (mirrors Schooling.nls and component files) ────────────────────────
# Applies BOIDS rules in order:
#   find-nearest-neighbor → separate / (cohere + align) → adjust-speed
# Returns list(heading, speed).
school_agent <- function(agent_id, agents) {
  focal <- agents[agents$id == agent_id, ]
  mates <- find_schoolmates(agent_id, agents)

  if (nrow(mates) == 0)
    return(list(heading = focal$heading, speed = focal$speed))

  # find-nearest-neighbor
  dists   <- sqrt((mates$x - focal$x)^2 + (mates$y - focal$y)^2)
  nn_dist <- min(dists)
  nn_x    <- mates$x[which.min(dists)]

  new_heading <- focal$heading
  new_speed   <- focal$speed

  if (nn_dist < focal$minimum_separation) {
    # separate: slow down if neighbor is ahead, turn away otherwise
    if (nn_x > focal$x) {
      new_speed <- max(focal$min_speed, focal$speed - 0.1)
    } else {
      new_heading <- focal$heading +
        (focal$separate_coefficient / 100) * sign(focal$x - nn_x) * 10
    }
  } else {
    # cohere: turn toward centroid of schoolmates
    centroid_x  <- mean(mates$x)
    new_heading <- focal$heading +
      (focal$cohere_coefficient / 100) * sign(centroid_x - focal$x) * 10

    # align: match mean heading of schoolmates
    mean_heading <- mean(mates$heading)
    new_heading  <- new_heading +
      (focal$align_coefficient / 100) * (mean_heading - new_heading)
  }

  # adjust-speed: speed up if a schoolmate is faster, otherwise slow down
  if (max(mates$speed) > focal$speed + 0.1) {
    new_speed <- focal$speed * 1.1
  } else if (focal$speed > focal$min_speed) {
    new_speed <- focal$speed * 0.9
  }
  new_speed <- max(new_speed, focal$min_speed)

  list(heading = new_heading %% 360, speed = new_speed)
}


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

log_ticks <- vector("list", N_TICKS)

for (tick in 1:N_TICKS) {

  # Step 1: migrate — face home and step forward
  migrating <- agents$start_migration &
               agents$landward_migration &
               !agents$at_destination

  move_results <- lapply(which(migrating), function(i) {
    migrate_landward(agents$x[i], agents$y[i],
                     agents$speed[i], agents$home_x[i])
  })

  agents$x[migrating]       <- sapply(move_results, `[[`, "x")
  agents$y[migrating]       <- sapply(move_results, `[[`, "y")
  agents$heading[migrating] <- sapply(move_results, `[[`, "heading")

  # Step 2: school — separation, cohesion, alignment, speed adjustment
  school_results <- lapply(agents$id, function(aid) school_agent(aid, agents))
  agents$heading <- sapply(school_results, `[[`, "heading")
  agents$speed   <- sapply(school_results, `[[`, "speed")

  # Step 3: check arrival
  agents <- agents |>
    mutate(
      at_destination     = x >= home_x,
      landward_migration = !at_destination
    )

  log_ticks[[tick]] <- tibble(
    tick       = tick,
    mean_x     = mean(agents$x),
    mean_speed = mean(agents$speed),
    n_arrived  = sum(agents$at_destination)
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

p1 <- ggplot(results, aes(x = tick, y = mean_x)) +
  geom_line(color = "#2c7bb6", linewidth = 1) +
  labs(title = "Mean agent position over time",
       x = "Tick", y = "Mean upstream position") +
  theme_minimal()

p2 <- ggplot(results, aes(x = tick, y = mean_speed)) +
  geom_line(color = "#d7191c", linewidth = 1) +
  labs(title = "Mean agent speed over time",
       x = "Tick", y = "Mean speed") +
  theme_minimal()

p3 <- ggplot(results, aes(x = tick, y = n_arrived)) +
  geom_step(color = "#1a9641", linewidth = 1) +
  labs(title = "Cumulative arrivals at home patch",
       x = "Tick", y = "Number of agents arrived") +
  theme_minimal()

library(patchwork)
combined <- p1 + p2 + p3 +
  plot_annotation(title = "Chapter 15: Schooling & Landward Migration")

ggsave("E:/Sample_ABM_Tutorials/Simple_Example_01/R/ch15_output.png", combined, width = 12, height = 4, dpi = 150)
message("Plot saved to ch15_output.png")

# Summary
message("\n── Summary ──────────────────────────────────────────")
message("Ticks run:        ", nrow(results))
message("Agents arrived:   ", sum(agents$at_destination), " / ", N_AGENTS)
message("Mean final speed: ", round(mean(agents$speed), 3))
