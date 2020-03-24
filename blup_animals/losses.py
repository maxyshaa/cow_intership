"""Module contains the solution of mixed model equation."""


from blup_animals.matrices import mme_matrix


def mme_solution(mme_left, mme_right: np.ndarray) -> np.ndarray:
    """Returns estimated parameters of MME.
    Args:
        mme_left: Left part of MME.
        mme_right: Right part of MME.

    Returns:
        blup: Array with estimated parameters from given equation.
    """
    params = np.dot(np.linalg.pinv(mme_left), mme_right)
    blup = params.reshape(-1, 1)

    pass


