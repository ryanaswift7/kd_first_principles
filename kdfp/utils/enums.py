from enum import Enum

class ProjectorTraining(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RANDOMIZED = "randomized"

class StudentReduction(str, Enum):
    NARROW = "narrow"
    SHALLOW = "shallow"

class ProjectionDirection(str, Enum):
    UP = 'up'
    DOWN = 'down'

def safe_enum(enum_class, value):
    """Convert a value to an enum member, or return None if value is None."""
    if value is None:
        return None
    return enum_class(value)