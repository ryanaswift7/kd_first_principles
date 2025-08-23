from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Dict, Any
import json

class ConfigBase(ABC, BaseModel):
    """
    Abstract base class for all configuration objects.
    Provides standardized methods for configuration representation and comparison.
    """
    
    @abstractmethod
    def to_config_dict(self) -> Dict[str, Any]:
        """
        Return a standardized dictionary representation for configuration purposes.
        This should exclude heavy objects and only include lightweight configuration fields.
        """
        pass
    
    @property
    def config_json(self) -> str:
        """Property that returns JSON representation for hashing."""
        return json.dumps(self.to_config_dict(), sort_keys=True, default=str)
    
    def __hash__(self) -> int:
        """Make objects hashable based on their configuration."""
        return hash(self.config_json)
    
    def __eq__(self, other: object) -> bool:
        """Compare objects based on their configuration."""
        if not isinstance(other, ConfigBase):
            return False
        return self.config_json == other.config_json
    
    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid'  # Prevent extra fields