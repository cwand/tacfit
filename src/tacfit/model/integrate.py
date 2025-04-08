import numpy as np
import numpy.typing as npt
from typing import Callable
import scipy


def model(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        irf: Callable[[npt.NDArray[np.float64],
                       dict[str, float]],
                      npt.NDArray[np.float64]],
        **kwargs: float) -> npt.NDArray[np.float64]:

    """Solves the model
    """

    res = np.zeros_like(t_out)

    # Check if delay is a model parameter. In that case, shift the input time
    t_d = 0.0
    if '_delay' in kwargs:
        t_d = kwargs['_delay']
    input_time = t_in + t_d

    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # The integrand is infunc(tau) * IRF(ti-tau)
        def integrand(tau: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return (np.interp(tau, input_time, in_func, left=0.0) *
                    irf(ti - tau, **kwargs))  # type: ignore

        # We integrate each step of the input function separately
        j = 0
        while input_time[j+1] < ti:
            y = scipy.integrate.quad(integrand, input_time[j], input_time[j+1])
            res[i] += y[0]
            j = j + 1

        # add last step from t[j] to ti
        y = scipy.integrate.quad(integrand, input_time[j], ti)
        res[i] += y[0]

    return res
