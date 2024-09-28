from ..data_models.trait import *

class Image:
    def __init__(self, name, traits) -> None:
        self.traits: dict[Trait, float] = traits
        self.name = name
    
    def __repr__(self) -> str:
        return f"Name: {self.name}, Trait: {self.traits}"