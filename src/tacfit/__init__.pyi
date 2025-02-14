import numpy as np
import numpy.typing as npt
from typing import Optional, Callable

from tacfit import model

def load_table(path: str) -> dict[str, npt.NDArray[np.float64]]: ...

def plot_tac(time_data: npt.NDArray[np.float64],
             input_data: npt.NDArray[np.float64],
             tissue_data: npt.NDArray[np.float64],
             labels: Optional[dict[str, str]] = ...,
             output: Optional[str] = ...): ...

def fit_leastsq(time_data: npt.NDArray[np.float64],
                tissue_data: npt.NDArray[np.float64],
                input_data: npt.NDArray[np.float64],
                model: Callable[[npt.NDArray[np.float64],
                                 npt.NDArray[np.float64],
                                 dict[str, float]],
                                npt.NDArray[np.float64]],
                params: dict[str, dict[str, float]],
                labels: dict[str, str],
                tcut: int,
                output: Optional[str]) -> None: ...