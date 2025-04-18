import numpy as np
import numpy.typing as npt
from typing import Optional, Callable, Union

from tacfit import model

def load_table(path: str) -> dict[str, npt.NDArray[np.float64]]: ...

def plot_tac(time_data: npt.NDArray[np.float64],
             input_data: npt.NDArray[np.float64],
             tissue_data: npt.NDArray[np.float64],
             labels: Optional[dict[str, str]] = ...,
             output: Optional[str] = ...): ...

def create_corrected_input_function(orig_time_data: npt.NDArray[np.float64],
                                    orig_infunc_data: npt.NDArray[np.float64],
                                    delay: float = ...) \
        -> tuple[npt.NDArray[np.float64], ...]: ...

def fit_leastsq(time_data: npt.NDArray[np.float64],
                tissue_data: npt.NDArray[np.float64],
                input_data: npt.NDArray[np.float64],
                model: Callable[[npt.NDArray[np.float64],
                                 npt.NDArray[np.float64],
                                 dict[str, float]],
                                npt.NDArray[np.float64]],
                params: dict[str, dict[str, float]],
                labels: dict[str, str],
                tcut: Optional[Union[int, list[int]]] = ...,
                scut: Optional[float] = ...,
                delay: Optional[float] = ...,
                confint: bool = ...,
                output: Optional[str] = ...,
                progress: bool = ...) -> None: ...


def mc_sample(time_data: npt.NDArray[np.float64],
              tissue_data: npt.NDArray[np.float64],
              input_data: npt.NDArray[np.float64],
              labels: dict[str, str],
              model: Callable[[npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               dict[str, float]],
                              npt.NDArray[np.float64]],
              params: dict[str, dict[str, float]],
              error_model: str,
              nsteps: int,
              nwalkers: int,
              nworkers: int,
              burn: int,
              thin: int,
              tcut: Optional[int] = ...,
              scut: Optional[float] = ...,
              delay: Optional[float] = ...,
              progress: Optional[bool] = ...,
              output: Optional[str] = ...) -> None: ...
