import numpy as np
import numpy.typing as npt
from typing import Callable

def irf_delay(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_step2(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_stepconst(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_normconst(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_stepnorm(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def irf_norm2(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...

def model(t_in: npt.NDArray[np.float64],
          in_func: npt.NDArray[np.float64],
          t_out: npt.NDArray[np.float64],
          irf: Callable[[npt.NDArray[np.float64],
                         dict[str, float]],
                         npt.NDArray[np.float64]],
          **kwargs: float) -> npt.NDArray[np.float64]: ...

