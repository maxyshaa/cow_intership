"""Module contains functions to obtain matrices for the mixed model equation construction.
Henderson (1949) presented the mixed model equations, which solves for b and a. The mixed model equations (MME)
were derived by maximizing the joint probability density of y and a. In matrix notation MME can be expressed as:
[[ X'X, X'Z], [Z'X, Z'Z+A^(-1)*alpha]] @ [[b], [a]] = [[X'y], [Z'y]].
"""


import numpy as np
import pandas as pd


def make_xyz(df: pd.DataFrame,
             response_var: list,
             fixed_var: list = ['Sex']) -> np.ndarray:
    """Produce x, z, y matrices. x matrix is associated with the fixed effect parameters,
     z matrix is associated with random effect parameters, y is a vector of observations.

    Args:
        df: DataFrame with records of animals.
        response_var: Response variable.
        fixed_var: Fixed effects, defaults to Sex.
    Returns:
        x,y,z matrices as a NumPy array.
    Raises:
        Some error.
    """
    x = pd.get_dummies(df[fixed_var]).to_numpy()
    x = np.fliplr(x)
    n_animals = df.shape[0]  # to fix z the the same shape
    z = np.identity(n_animals)[1]
    y = df[[response_var]].to_numpy()

    return x, z, y


def rel_matrix(df_long: pd.DataFrame) -> None:
    """Returns an inverse relationship matrix. Relationship matrix (A) is expressed by Thompson as
    A = TDT', where T is a lower triangular matrix and D is a diagonal matrix.

    Args:
        df_long: Long format of pedigree.
    Returns:
        inverse_a: Inverse relationship matrix as array type.
    """
    pass


def mme_matrix(x, z, y, inverse_a: np.ndarray, sigma_sq_e,
               sigma_sq_a: int) -> np.ndarray:
    """Calculate matrices of the mixed model to design the least squares equations (LSE).

    Args:
        x: Matrix obtained in function make_xyz.
        z: Matrix obtained in function make_xyz.
        y: Matrix obtained in function make_xyz.
        inverse_a: Inverse relationship matrix obtained in function rel_matrix.
        sigma_sq_a: Variance of random animals effects.
        sigma_sq_e: Variance of random residual effects.
    Returns:
        mme_left, mme right: Matrices which form MME.

    """
    alpha = sigma_sq_e/sigma_sq_a

    x_matrix = np.concatenate((np.dot(x.T, x), np.dot(z.T, x)), axis=0)
    z_matrix = np.concatenate((np.dot(x.T, z), np.dot(z.T, z) + inverse_a*alpha), axis=0)

    mme_left = np.concatenate((x_matrix, z_matrix), axis=1)
    mme_right = np.concatenate((np.dot(x.T, y), np.dot(z.T, y)), axis=0)

    return mme_left, mme_right

