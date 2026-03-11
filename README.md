# F1 Race Track Agent Comparison

**Status:** In Progress / To Be Started Soon

## Overview
This project investigates how different intelligent agent architectures perform in optimising race performance under varying track and vehicle conditions. The environment consists of a single fixed F1 race track, which can be implemented using **Unity / Pygame / TORCS / CarRacing-v2 from Gymnasium**. The simulation includes dynamic environmental factors such as dry, wet, and drying track conditions, tyre compounds (Soft, Medium, Hard, Wet), a tyre degradation model, and a simple fuel weight model.

## Research Question
**How do different agent architectures perform in optimising race time under changing track grip and tyre degradation conditions?**

## Agents
Three autonomous agents will be compared:

1. **Rule-Based Agent**  
   - Follows a predefined racing line  
   - Pits after a fixed number of laps or tyre threshold  
   - Does not adapt to sudden changes in track conditions

2. **Utility-Based Agent**  
   - Follows a predefined racing line  
   - Makes dynamic decisions on throttle, speed, and pit timing based on current grip, track conditions, tyre wear, and remaining race distance/laps  
   - Adapts strategy to sudden weather changes (e.g., rain starting or track drying)

3. **Reinforcement Learning Agent**  
   - Learns optimal throttle, steering, and pit strategies through trial-and-error interactions  
   - Can adapt to dynamic track conditions, balancing lap times, tyre efficiency, and track violations

## Performance Metrics
The agents will be evaluated based on:

- Total race time  
- Average lap time  
- Lap time variance  
- Track violations  
- Tyre efficiency  

## Environment Options
The race track environment can be implemented in:  
**Unity / Pygame / TORCS / CarRacing-v2 from Gymnasium**  

## Project Goals
- Compare how different AI architectures handle dynamic race conditions  
- Analyse the trade-offs between safety, speed, and tyre management  
- Conduct controlled experiments to evaluate agent effectiveness under different track and weather scenarios  

## Future Work
- Implement simulation environment and agent behaviours  
- Train reinforcement learning agent  
- Run experiments for different track conditions and tyre strategies  
- Collect and visualise performance metrics  

