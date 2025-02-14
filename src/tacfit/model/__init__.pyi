import numpy as np
import numpy.typing as npt

def model_step_2(
        t: npt.NDArray[np.float64], in_func: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]: ...