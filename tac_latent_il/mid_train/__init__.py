from .TactileDynamicsBase import TactileDynamicsBase
from .TactileDynamicsInfoNCEBase import TactileDynamicsInfoNCEBase
from .TactileDynamicsMSEBase import TactileDynamicsMSEBase


MID_TRAIN_METHODS = {
    "TactileDynamicsInfoNCE": TactileDynamicsInfoNCEBase,
    "TactileDynamicsMSE": TactileDynamicsMSEBase,
}
