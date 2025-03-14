import numpy as np
import numpy.typing as npt
import scipy


def _split_arrays(t_in: npt.NDArray[np.float64],
                  in_func: npt.NDArray[np.float64],
                  t: float,
                  tc1: float,
                  tc2: float) -> tuple[npt.NDArray[np.float64], ...]:
    # Split a time array and a function array into two separate arrays.
    # Given the arrays
    #   t_in = [t_0, t_1, ..., t_n]
    #   in_func = [f_0, f_1, ..., f_n]
    # this function returns the arrays:
    # - [tx1, t_k, t_k+1, ..., t_l, tx2]
    # - [f*(tx1), f_k, f_k+1, ..., f_l, f*(tx2)]
    # - [tx2, t_l+1, t_l+2, ..., t_m, t]
    # - [f*(tx2), f_l+1, f_l+2, ..., f_m, f*(t)]
    # where tx1 = max(0,t - tc1), tx2 = max(0,t - tc2), t_k is the smallest
    # element in t_in larger than tx1 (if any), tl is the largest element
    # in t_in smaller than tx2 (if any), t_m is the largest element in t_in
    # smaller than t (if any) and f*(x) indicates the interpolated value of
    # in_func at x.

    tx1 = max(0.0, t - tc1)
    tx2 = max(0.0, t - tc2)
    arr1 = t_in[np.logical_and(t_in > tx1, t_in < tx2)]
    arr1 = np.append([tx1], arr1)
    arr1 = np.append(arr1, tx2)

    arr2 = in_func[np.logical_and(t_in > tx1, t_in < tx2)]
    arr2 = np.append([np.interp(tx1, t_in, in_func, left=0.0)], arr2)
    arr2 = np.append(arr2, np.interp(tx2, t_in, in_func, left=0.0))

    arr3 = t_in[np.logical_and(t_in > tx2, t_in < t)]
    arr3 = np.append([tx2], arr3)
    arr3 = np.append(arr3, t)

    arr4 = in_func[np.logical_and(t_in > tx2, t_in < t)]
    arr4 = np.append([np.interp(tx2, t_in, in_func, left=0.0)], arr4)
    arr4 = np.append(arr4, np.interp(t, t_in, in_func, left=0.0))

    return arr1, arr2, arr3, arr4


def model_step2(t_in: npt.NDArray[np.float64],
                in_func: npt.NDArray[np.float64],
                t_out: npt.NDArray[np.float64],
                **kwargs: float) -> npt.NDArray[np.float64]:
    """Solves the model where the input response function is assumed to be a
    2-step function.
    This function calculates the convolution of a sampled input function with
    a 2-step function. The step function has value amp1 on the interval
    [0, extent1), value amp2 on the interval [extent1, extent2) and value 0 on
    the interval [extent2, infinity).
    The convolution is evaluated at the time points given in t_out and
    returned as a list.
    The convolution is performed numerically using scipy.integrate.quad and
    the input function is interpolated linearly between sample points.

    Arguments:
    t_in    --  The time points of the input function samples.
    in_func --  The input function samples.
    t_out   --  The time points where the model should be evaluated.
    amp1    --  The amplitude of the step function on [0, extent1).
    extent1 --  The length of the first step function.
    amp2    --  The amplitude of the step function on [0, extent2).
    extent2 --  The length of the second step function.

    Return value:
    A list containing the modeled values at each time point.
    """

    res = np.zeros_like(t_out)

    # Unpack parameters
    ext1 = kwargs['extent1']
    ext2 = kwargs['extent2']
    amp1 = kwargs['amp1']
    amp2 = kwargs['amp2']

    # The function to be integrated is just the amplitudes times the integral
    # of the input function. The amplitudes will be multiplied after
    # integration.
    def integrand(tau: float):
        return np.interp(tau, t_in, in_func, left=0.0)

    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # We split the integral up into the two step-functions and increase
        # the maximum subdivision limit to avoid discontinuity problems

        # We slacken the error tolerance for faster fits and to avoid warnings
        # about roundoff errors.

        # First integral from max(0, t-ext2) to max(0, t-ext1)
        y1 = scipy.integrate.quad(integrand,
                                  max(0.0, ti - ext2),
                                  max(0.0, ti - ext1),
                                  limit=500,
                                  epsabs=0.001,
                                  epsrel=0.001)

        # Second integral from max(0, t-ext1) to t
        y2 = scipy.integrate.quad(integrand,
                                  max(0.0, ti - ext1),
                                  ti,
                                  limit=500,
                                  epsabs=0.001,
                                  epsrel=0.001)

        # Multiply by the amplitudes
        res[i] = amp2*y1[0] + amp1*y2[0]

    return res
