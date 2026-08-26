"""
testloader.py

Simple script used to verify that the data loaders work correctly.
"""

from pathlib import Path
from config import DATA_DIR, LABEL_FILE, SIM7200_DIR
from src.loader import SimulationLoader
from src.labels import LabelLoader


# ------------------------------------------------------------------
# Define data locations
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# Create loader objects
# ------------------------------------------------------------------

loader = SimulationLoader(SIM7200_DIR)

labels = LabelLoader(LABEL_FILE)


# ------------------------------------------------------------------
# Print basic information
# ------------------------------------------------------------------

print("Simulation directory:")
print(DATA_DIR.resolve())

print("\nNumber of simulations:")
print(len(loader.sim_ids()))

print("\nFirst ten simulation IDs:")
print(loader.sim_ids()[:10])


# ------------------------------------------------------------------
# Load one simulation
# ------------------------------------------------------------------

simid = 0

simulation = loader.load_simulation(simid)

print("\nSimulation ID:")
print(simulation["id"])


print("\nBalance file:")

print(simulation["balance"].head())


print("\nTemperature file:")

print(simulation["temperature"].head())


# ------------------------------------------------------------------
# Print labels
# ------------------------------------------------------------------

print("\nInitial conclusion (7200 s):")

print(labels.get_initial_conclusion(simid))


print("\nFinal conclusion (35000 s):")

print(labels.get_label(simid))