import math

import numpy as np
import numpy.typing as npt
import scipy

def model_normconst(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:
    """Solves the model where the impulse response function starts at the value
    amp1, transitions smoothly to a new value amp2 around the time point
    extent1, and then stays constant at amp2.
    The smoothness is implemented as a cumulative normal distribution function
    with mean extent1 and standard deviation width1.
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

    # Unpack parameters
    ext1 = kwargs['extent1']
    amp1 = kwargs['amp1']
    wid1 = kwargs['width1']
    amp2 = kwargs['amp2']


    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # We have to integrate the input function from t=0 to t=ti:
        t_in_cut = [0.0]
        t_in_cut = np.append(t_in_cut, t_in[t_in < ti])
        t_in_cut = np.append(t_in_cut, ti)

        f_in_cut = [0.0]
        f_in_cut = np.append(f_in_cut, in_func[t_in < ti])
        f_in_cut = np.append(f_in_cut, np.interp(ti, t_in, in_func))

        irf_t = ((ti - t_in_cut) - ext1)/(math.sqrt(2.0) * wid1)
        cdf = 0.5 * (1.0 + scipy.special.erf(irf_t))
        irf = (amp2 + (amp1 - amp2) * (1.0 - cdf))

        res[i] = scipy.integrate.trapezoid(irf*f_in_cut, t_in_cut)
        '''
        # Define integrand at this timepoint
        def integrand(tau: float) -> float:
            # cdf = scipy.stats.norm.cdf(ti - tau, loc=ext1, scale=wid1)
            z = ((ti - tau) - ext1)/(math.sqrt(2.0) * wid1)
            cdf = 0.5 * (1.0 + scipy.special.erf(z))
            infunc_val = np.interp(tau, t_in, in_func, left=0.0)
            return (amp2 + (amp1 - amp2) * (1.0 - cdf)) * infunc_val

        # To avoid discontinuity problems we do the integral as a sum of
        # integrals between the time points on the sampled IRF.
        for k in range(0, t_in_cut.size - 1):
            y = scipy.integrate.quad(integrand, t_in_cut[k], t_in_cut[k+1],
                                     epsrel=0.001,
                                     epsabs=0.001)
            res[i] += y[0]
        #y = scipy.integrate.quad(integrand, 0.0, ti)
        #res[i] = y[0]'''


    return res