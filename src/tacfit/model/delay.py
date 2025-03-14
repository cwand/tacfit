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

    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # We use trapzeoids to calculate integral of input, but first we need
        # to split the array
        t = [0.0]
        t = np.append(t, input_time_dly[input_time_dly < ti])
        t = np.append(t, ti)

        f = [0.0]
        f = np.append(f, in_func[input_time_dly < ti])
        f = np.append(f, np.interp(ti, input_time_dly, in_func, left=0.0))

        y1 = scipy.integrate.trapezoid(f, t)

        # Multiply by the amplitudes
        res[i] = k * y1

    return res
