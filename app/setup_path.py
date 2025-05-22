import sys
import os

# Add AirSim Python API path
airsim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AirSim/PythonClient")
if airsim_path not in sys.path:
    sys.path.append(airsim_path)