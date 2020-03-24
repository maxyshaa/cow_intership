"""This module contains functions that get the data to work with."""

import numpy as np
import pandas as pd


def get_pedigree(name: str) -> pd.DataFrame:
    """Obtain DataFrame which is basically pedigree but also contains information about animal traits.

    Args:
        name: Name of the csv table.
    Returns:
        df: Returns DataFrame.
    """
    df = pd.read_csv(name)

    pass


def get_long_pedigree(df: pd.DataFrame) -> pd.DataFrame:
    """Obtained pedigree redesign a little. Add parents to top of pedigree.

    Args:
        df: DataFrame obtained above.

    Returns:
        df_long: Long format of given pedigree.
    """
    return None


