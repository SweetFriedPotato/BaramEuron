from .sklearn_models import SklearnModel
from .mlp import MLPNet
from .gru import GRUNet

MODEL_REGISTRY={"random_forest":SklearnModel,"extra_trees":SklearnModel,"mlp":MLPNet,"gru":GRUNet}

