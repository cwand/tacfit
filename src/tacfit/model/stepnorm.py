import math

import numpy as np
import numpy.typing as npt
import scipy


def irf_stepnorm(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:
    """Computes the IRF of the stepnorm model, where the impulse response
    function starts at the value amp1, drops to a new value amp2 at time
    extent1 and then transitions smoothly to 0 at around time extent2.
    The smoothness is implemented as a cumulative normal distribution function
    with mean extent2 and standard deviation width2.
    NOTE: The cumulative normal distribution is not equal to 1 at t=0, so if
    the mean is small and/or the width is large, this will mean the IRF will
    start at a smaller value than amp1.

    Arguments:
    t       --  Time points where the IRF should be computed.
    amp1    --  The value of the IRF at t=0
    extent1 --  The midpoint of the transition from amp1 to amp2
    width1  --  The width of the transition between amp1 and amp2
    amp2    --  The value of the IRF at t=inf.

    Return value:
    A list containing the modeled values at each time point.
    """

    amp1 = kwargs['amp1']
    amp2 = kwargs['amp2']
    ext1 = kwargs['extent1']
    ext2 = kwargs['extent2']
    wid2 = kwargs['width2']

    # Calculate normcdf using error function (seems to be much quicker)
    tt = (t - ext2) / (math.sqrt(2.0) * wid2)
    cdf = 0.5 * (1.0 + scipy.special.erf(tt))

    # Calculate input response function
    res = amp2 * (1.0 - cdf) + (amp1 - amp2) * (t < ext1)
    return np.array(res)


def _split_arrays(t_in: npt.NDArray[np.float64],
                  in_func: npt.NDArray[np.float64],
                  t: float,
                  tc: float) -> tuple[npt.NDArray[np.float64], ...]:
    # Split a time array and a function array into two separate arrays.
    # Given the arrays
    #   t_in = [t_0, t_1, ..., t_n]
    #   in_func = [f_0, f_1, ..., f_n]
    # this function returns the arrays:
    # - [0, t_0, t_1, ..., t_k, tx]
    # - [f*(0), f_0, f_1, ..., f*(tx)]
    # - [tx, t_k+1, t_k+2, ..., t_l, t]
    # - [f*(tx), f_k+1, f_k+2, ..., f_l, f*(t)]
    # where tx = max(0,t - tc), t_k is the largest element in t_in smaller
    # than tx (if any), tl is the largest element in t_in smaller than t (if
    # any) and f*(x) indicates the interpolated value of in_func at x.

    tx = max(0.0, t - tc)
    arr1 = t_in[t_in < tx]
    arr1 = np.append([0.0], arr1)
    arr1 = np.append(arr1, tx)

    arr2 = in_func[t_in < tx]
    arr2 = np.append([0.0], arr2)
    arr2 = np.append(arr2, np.interp(tx, t_in, in_func, left=0.0))

    arr3 = t_in[np.logical_and(t_in > tx, t_in < t)]
    arr3 = np.append([tx], arr3)
    arr3 = np.append(arr3, t)

    arr4 = in_func[np.logical_and(t_in > tx, t_in < t)]
    arr4 = np.append([np.interp(tx, t_in, in_func, left=0.0)], arr4)
    arr4 = np.append(arr4, np.interp(t, t_in, in_func, left=0.0))

    return arr1, arr2, arr3, arr4


def model_stepnorm(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:
    """Solves the model where the impulse response function starts at the value
    amp1, drops to a new value amp2 at time extent1 and then transitions
    smoothly to 0 at around time extent2.
    The smoothness is implemented as a cumulative normal distribution function
    with mean extent2 and standard deviation width2.
    NOTE: The cumulative normal distribution is not equal to 1 at t=0, so if
    the mean is small and/or the width is large, this will mean the IRF will
    start at a smaller value than amp1.
    This function calculates the convolution of a sampled input function with
    the impulse response function.
    The convolution is evaluated at the time points given in t_out and
    returned as a list.

    Arguments:
    t_in    --  The time points of the input function samples.
    in_func --  The input function samples.
    t_out   --  The time points where the model should be evaluated.
    amp1    --  The value of the IRF at t=0
    extent1 --  The midpoint of the transition from amp1 to amp2
    width1  --  The width of the transition between amp1 and amp2
    amp2    --  The value of the IRF at t=inf.

    Return value:
    A list containing the modeled values at each time point.
    """

    res = np.zeros_like(t_out)

    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # The integrand is infunc(tau) * IRF(ti-tau)
        def integrand(tau: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return (np.interp(tau, t_in, in_func, left=0.0) *
                    irf_stepnorm(ti - tau, **kwargs))

        # We integrate each step of the input function separately
        j = 0
        while t_in[j + 1] < ti:
            y = scipy.integrate.quad(integrand, t_in[j], t_in[j + 1])
            res[i] += y[0]
            j = j + 1

        # add last step from t[j] to ti
        y = scipy.integrate.quad(integrand, t_in[j], ti)
        res[i] += y[0]

    return res
