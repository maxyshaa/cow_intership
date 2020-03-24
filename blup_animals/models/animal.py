"""Module contains models computation."""


import numpy as np
from blup_animals.losses import mme_solution
from blup_animals.lmatrices import mme_matrix


def animal_model(x, z, y, inverse_a: np.ndarray,
                 sigma_sq_a: int) -> pd.DataFrame:
    """Model computation.

    Args:
        x: Numpy array of independent variables.
        z:
        y: Numpy array of dependent variable.


    Returns:
         results_df: DataFrame with estimated parameters.
    """

#    mme_left, mme_right = mme_matrix
#    calling functions mme_solution
#    animals = df_long.Calf.to_numpy()
#    mme_left_diag = np.linalg.inv(mme_left).diagonal()[-Z.T.shape[0]:]
    r_squared = 1 - mme_left_diag * alpha
    r = np.sqrt(r_squared)
    sep = np.sqrt((1 - r_squared) * sigma_sq_a)

    results_df = pd.DataFrame({'Animal': animals, 'BLUP': blup,
                               'r_squared': r_squared, 'r': r, 'SEP': sep})

    pass
