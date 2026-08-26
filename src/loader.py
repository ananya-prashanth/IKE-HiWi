
"""
loader.py

This module loads the simulation output files.

Each simulation consists of:
1. A balance file (.xlsx)
2. A particle temperature file (.csv)

Example:
00042_balance_readable.xlsx
00042.00042.temp_unwrapped.csv
"""

from pathlib import Path
import pandas as pd


class SimulationLoader:

    """
    Loads simulation data from a directory.

    Parameters
    ----------
    sim_dir : str or Path
        Directory containing the simulation files.
    """

    def __init__(self, sim_dir):
        self.sim_dir= Path(sim_dir)

    def sim_ids(self):
        """
        Return a sorted list of all available simulation IDs.

        Example
        -------
        [0, 1, 2, ..., 999]
        """

        files= self.sim_dir.glob("*_balance_readable.xlsx")

        ids=sorted (
            int(f.name[:5])
            for f in files
        )

        return ids
    
    def load_balance(self, simid):

        """
        Load the balance file for one simulation.

        Parameters
        ----------
        simid : int

        Returns
        -------
        pandas.DataFrame
        """

        filename= f"{simid:05d}_balance_readable.xlsx"

        filepath = self.sim_dir / filename

        print(filepath)

        return pd.read_excel(filepath)

    def load_temp(self,simid):

        """
        Load the particle temperature file for one simulation.

        Parameters
        ----------
        simid : int

        Returns
        -------
        pandas.DataFrame
        """

        filename= f"{simid:05d}.{simid:05d}.temp_unwrapped.csv"

        return pd.read_csv(
            self.sim_dir / filename
        )

    def load_simulation(self, simid):

        """
        Load both files belonging to one simulation.

        Parameters
        ----------
        simid : int

        Returns
        -------
        dict
            Dictionary containing:
                id
                balance dataframe
                temperature dataframe
        """


        return{
            "id": simid,
            "balance": self.load_balance(simid),
            "temperature": self.load_temp(simid)
        }

