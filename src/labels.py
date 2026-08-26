"""
labels.py

Loads the Excel file containing:

- Input parameters
- 7200 second conclusion
- 35000 second conclusion

The simulation number is used as the index so that
labels can be retrieved efficiently.
"""

from pathlib import Path
import pandas as pd

class LabelLoader:

    """
    Loads the simulation labels and input parameters.

    Parameters
    ----------
    excel_file : str or Path
        Path to the Excel file containing labels.
    """
    

    def __init__(self, excel_file):
        self.excel_file = Path(excel_file)
        self.df= pd.read_excel(self.excel_file)

        # index by sim nr.
        self.df.set_index("Simulation No", inplace=True)
    

    def data(self):
        """
        Return the complete label table.

        Returns
        -------
        pandas.DataFrame
        """

        return self.df

    def get_row(self, simid):

        """
        Return the complete row corresponding to one simulation.

        Parameters
        ----------
        simid : int

        Returns
        -------
        pandas.Series
        """

        return self.df.loc[simid]

    def get_label(self, simid):

        """
        Return the final conclusion after the timed-out simulations
        were extended to 35000 seconds.

        Returns
        -------
        str
        """

        return self.df.loc[simid, "Timed Out Conclusion"]

    def get_initial_conclusion(self, simid):

        """
        Return the conclusion after the original 7200 second simulation.

        Returns
        -------
        str
        """
        
        return self.df.loc[simid, "Conclusion"]

    