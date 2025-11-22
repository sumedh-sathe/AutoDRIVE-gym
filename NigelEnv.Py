import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import queue
import socketio
import eventlet
from flask import Flask
import json
import autodrive


class NigelEnv(gym.Env):
    """
    A Gymnasium Environment for the AutoDrive Simulator.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(NigelEnv, self).__init__()

        # -----------------------------------------------------------------
        # 1. Define Action Space
        # -----------------------------------------------------------------
        # continuous actions: Steering [-1, 1] and Throttle [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

        # -----------------------------------------------------------------
        # 2. Define Observation Space
        # -----------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # -----------------------------------------------------------------
        # 3. Initialize Communication Queues and Vehicle
        # -----------------------------------------------------------------
        self.input_queue = queue.Queue(maxsize=1)  # Data from Sim -> Gym
        self.output_queue = queue.Queue(maxsize=1) # Action from Gym -> Sim
        
        self.nigel_1 = autodrive.Nigel()
        self.nigel_1.id = 'V1'
        
        # -----------------------------------------------------------------
        # 4. Start Server in a Background Thread
        # -----------------------------------------------------------------
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True # Kills thread when main program exits
        self.server_thread.start()
        
        print("Waiting for simulator connection...")
        # We assume the first message sent by the sim is the initial state
        # Logic to wait for connection is handled in reset()

    def step(self, action):
        """
        1. Send action to the server thread (which sends it to the Simulator).
        2. Wait for new state from the Simulator.
        3. Calculate reward and done flag.
        """
        # 1. Put the action in the queue for the Server Thread to pick up
        try:
            self.output_queue.put(action, timeout=1.0)
        except queue.Full:
            print("Error: Simulator is not accepting actions (Queue Full).")

        # 2. Wait for the observation (blocking wait)
        try:
            # Wait up to 5 seconds for the simulator to respond
            # If the sim is paused or crashed, this prevents the Gym loop from hanging forever
            data = self.input_queue.get(timeout=5.0)
        except queue.Empty:
            print("Error: Simulator timed out.")
            return np.zeros(self.observation_space.shape), 0, True, False, {}

        # 3. Process Observation
        observation = self._process_data(data)

        # 4. Calculate Reward (IMPLEMENT YOUR LOGIC HERE)
        # Example: Reward for speed, penalty for going off-road
        reward = 0.0 
        
        # 5. Check Termination (IMPLEMENT YOUR LOGIC HERE)
        # Example: Collision detected
        terminated = False 
        truncated = False
        
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Clear queues to ensure fresh start
        with self.input_queue.mutex:
            self.input_queue.queue.clear()
        with self.output_queue.mutex:
            self.output_queue.queue.clear()

        # Send a reset command to the vehicle logic
        # We push a special flag or specific action that triggers reset
        # Since the server loop waits for an action, we send a "reset action"
        reset_action = np.array([0.0, 0.0]) # Neutral action
        self.nigel_1.reset_command = True # Set the internal flag
        
        self.output_queue.put(reset_action)
        
        # Wait for the first frame after reset
        try:
            data = self.input_queue.get(timeout=10.0)
            self.nigel_1.reset_command = False # Toggle flag back off
        except queue.Empty:
            print("Error: Simulator did not connect or reset in time.")
            return np.zeros(self.observation_space.shape), {}

        observation = self._process_data(data)
        return observation, {}

    def _process_data(self, data):
        """
        Convert raw JSON/Dict data from simulator into numpy array.
        """
        # self.nigel_1.parse_data(data, verbose=False) <-- Already called in thread
        # Extract features relevant to RL agent here
        # Return dummy zeros matching observation space for now
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _run_server(self):
        """
        This runs in a separate thread. It hosts the Flask/SocketIO server.
        """
        sio = socketio.Server()
        app = Flask(__name__)

        @sio.on('connect')
        def connect(sid, environ):
            print(f'Client Connected: {sid}')

        @sio.on('Bridge')
        def bridge(sid, data):
            if data:
                # --- 1. PERCEPTION (Received from Sim) ---
                self.nigel_1.parse_data(data, verbose=False)
                
                # Send this data to the main Gym Thread
                # We use put_nowait to avoid blocking the network thread if Gym is slow
                # but in strict sync mode, we might want to block.
                try:
                    self.input_queue.put(data, block=True, timeout=5)
                except queue.Full:
                    pass # Drop frame if Gym is processing too slowly

                # --- 2. WAIT FOR ACTION (From Gym Agent) ---
                try:
                    # Block until the Agent calls step() and provides an action
                    action = self.output_queue.get(block=True, timeout=10)
                except queue.Empty:
                    # Fail-safe: if Gym dies or hangs, apply brakes
                    action = [0.0, 0.0] 

                # --- 3. CONTROL (Apply to Vehicle) ---
                throttle = float(action[1])
                steering = float(action[0])

                self.nigel_1.cosim_mode = 0
                self.nigel_1.throttle_command = throttle
                self.nigel_1.steering_command = steering
                
                # Fixed commands (can be added to action space if needed)
                self.nigel_1.headlights_command = 0 
                self.nigel_1.indicators_command = 0 
                
                # Note: reset_command is handled in the reset() method logic

                # Generate message
                json_msg = self.nigel_1.generate_commands(verbose=False)

                try:
                    sio.emit('Bridge', data=json_msg)
                except Exception as e:
                    print(f"Emit Error: {e}")

        # Wrap and start server
        app = socketio.Middleware(sio, app)
        
        # Suppress server logs for cleaner Gym output
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app, log_output=False)

if __name__ == "__main__":
    # Example usage
    env = NigelEnv()
    
    print("Environment created. Resetting...")
    obs, info = env.reset()
    print("Reset complete. Starting loop...")

    for i in range(100):
        # Sample random action: [Steering, Throttle]
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i}: Action={action} | Reward={reward}")
        
        if terminated or truncated:
            obs, info = env.reset()

    print("Done.")