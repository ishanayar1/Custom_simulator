# Custom_simulator

The project implements a custom railway simulator integrated with Deep Q-Network (DQN) reinforcement learning to optimize train scheduling and passenger delay in railway networks, including real-world scenarios like the Chennai Metro system.​

The simulator models train transitions, passenger boarding/deboarding, and line switching at junctions, with dynamic event triggers and a signaling system that adapts halt times based on passenger queues and train capacity.​

Key classes include Train, Section, Junction, Line, and Passenger, capturing the network’s operational states and activities.​

Setup & Usage Instructions
The codebase is organized in two main folders:

CustomSimulator: Contains the core railway network simulation logic.

RLIntegration: Implements DQN-based reinforcement learning applied to the simulator environment.​

User Inputs required:

Network description file specifying trains (name, line, start time, speed, capacity), junctions (name, line, signal status, coordinates, passenger arrival rates), and sections connecting junctions.

Parameters for RL training, such as episode count and DQN hyperparameters (epsilon, gamma, learning rate).

Outputs generated:

Train time-tables with details on arrivals and departures at junctions and sections.

Final passenger list showing journey details and computed delays per passenger.

Reward logs (negative passenger delays per episode) to track RL performance improvements.

The DQN agent trains over multiple episodes, learning to minimize overall passenger delay by making adaptive decisions at junctions (e.g., when to wait for transferring passengers versus proceeding).​

Output shows fluctuating (negative) reward values as the RL agent seeks improved passenger outcomes; tuning of RL parameters like epsilon and gamma can further refine results.

The custom simulator and RL framework successfully model complex scenarios, such as intersecting railway lines and signal-based scheduling, and provide insights applicable to real-world metro networks.​



