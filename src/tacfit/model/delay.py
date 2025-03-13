import numpy as np
import numpy.typing as npt
import scipy


def model_delay(t_in: npt.NDArray[np.float64],
                in_func: npt.NDArray[np.float64],
                t_out: npt.NDArray[np.float64],
                **kwargs: float) -> npt.NDArray[np.float64]:
    """Solves the model where the input function is delayed and the impulse
    response function is simply a constant.
    This function calculates the integral of the delayed input function (i.e.
    t_in is shifted by some amount) and multiplies it by a constant.
    The integral is evaluated at the time points given in t_out and
    returned as a list.
    The integration is performed numerically using scipy.integrate.quad and
    the input function is interpolated linearly between sample points.

    Arguments:
    t_in    --  The time points of the input function samples.
    in_func --  The input function samples.
    t_out   --  The time points where the model should be evaluated.
    k       --  The constant multiplied on to the integral.
    delay   --  The shift of the input function time data before integration.

    Return value:
    A list containing the modeled values at each time point.
    """

    res = np.zeros_like(t_out)

    # Unpack parameters
    k = kwargs['k']
    delay = kwargs['delay']

    # Make delayed input function time data
    input_time_dly = t_in + delay

    # The function to be integrated is just the constant times the integral
    # of the input function. The amplitudes will be multiplied after
    # integration. We assume the input function is 0 before the first time
    # point
    def integrand(tau: float):
        return np.interp(tau, input_time_dly, in_func, left=0)

    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # Calculate integral
        y1 = scipy.integrate.quad(integrand, 0, ti)

        # Multiply by the amplitudes
        res[i] = k * y1[0]

    return res
