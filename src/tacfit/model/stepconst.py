import numpy.typing as npt
import numpy as np
import scipy


def irf_stepconst(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:

    amp1 = kwargs['amp1']
    amp2 = kwargs['amp2']
    ext1 = kwargs['extent1']

    # Calculate normcdf using error function (seems to be much quicker)
    res = amp2 * np.ones_like(t)
    for i in range(len(t)):
        if t[i] < ext1:
            res[i] = amp1
    return res

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


def model_stepconst(t_in: npt.NDArray[np.float64],
                    in_func: npt.NDArray[np.float64],
                    t_out: npt.NDArray[np.float64],
                    **kwargs: float) -> npt.NDArray[np.float64]:
    """Solves the model where the impulse response function is assumed to be a
    step function followed by a constant.
    This function calculates the convolution of a sampled input function with
    a step-const-function. The step function has value amp1 on the interval
    [0, extent1), followed by the value amp2 on the interval
    [extent1, infinity).
    The convolution is evaluated at the time points given in t_out and
    returned as a list.
    The convolution is performed numerically using scipy.integrate.trapezoid
    and the input function is interpolated linearly between sample points.

    Arguments:
    t_in    --  The time points of the input function samples.
    in_func --  The input function samples.
    t_out   --  The time points where the model should be evaluated.
    amp1    --  The amplitude of the step function on [0, extent1).
    extent1 --  The length of the first step function.
    amp2    --  The value of the response function on [extent1, infinity).

    Return value:
    A list containing the modeled values at each time point.
    """

    res = np.zeros_like(t_out)

    # Unpack parameters
    ext1 = kwargs['extent1']
    amp1 = kwargs['amp1']
    amp2 = kwargs['amp2']

    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # We split the integral up at the discontinuity point at t=ti - extent1
        t1, f1, t2, f2 = _split_arrays(t_in, in_func, ti, ext1)
        y1 = scipy.integrate.trapezoid(f1, t1)
        y2 = scipy.integrate.trapezoid(f2, t2)

        # Multiply by the amplitudes
        res[i] = amp2 * y1 + amp1 * y2

    return res
