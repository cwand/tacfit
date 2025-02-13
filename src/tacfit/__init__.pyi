import numpy as np
import numpy.typing as npt
from typing import Optional

def load_table(path: str) -> dict[str, npt.NDArray[np.float64]]: ...

def plot_data_nofit(time_data: npt.NDArray[np.float64],
                    inp_data: npt.NDArray[np.float64],
                    tis_data: npt.NDArray[np.float64],
                    labels: Optional[dict[str, str]] = ...,
                    out_file: Optional[str] = ...): ...
