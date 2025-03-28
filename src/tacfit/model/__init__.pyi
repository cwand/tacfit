import numpy as np
import numpy.typing as npt

def model_delay(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_delay(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def model_step2(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_step2(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def model_stepconst(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_stepconst(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def model_normconst(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_normconst(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def model_stepnorm(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_stepnorm(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...