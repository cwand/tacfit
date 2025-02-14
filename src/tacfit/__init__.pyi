import numpy as np
import numpy.typing as npt
from typing import Optional

# From core.py

def load_table(path: str) -> dict[str, npt.NDArray[np.float64]]: ...

def plot_tac(time_data: npt.NDArray[np.float64],
             input_data: npt.NDArray[np.float64],
             tissue_data: npt.NDArray[np.float64],
             labels: Optional[dict[str, str]] = ...,
             output: Optional[str] = ...): ...