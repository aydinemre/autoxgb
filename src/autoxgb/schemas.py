from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

from .enums import ProblemType


class ModelConfig(BaseModel):
    train_filename: Optional[str] = None
    train_df: Optional[pd.DataFrame] = None
    test_filename: Optional[str] = None
    test_df: Optional[pd.DataFrame] = None
    idx: str
    targets: List[str]
    problem_type: ProblemType
    output: str
    features: List[str]
    num_folds: int
    use_gpu: bool
    seed: int
    categorical_features: List[str]
    num_trials: int
    time_limit: Optional[int] = None
    fast: bool

    class Config:
        arbitrary_types_allowed = True
