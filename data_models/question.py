from abc import ABC
from enum import Enum
from ..data_models.trait import *

class QuestionType(Enum):
    Trait = "Trait"
    Guess = "Guess"

class Question(ABC):
    def __init__(self, type: QuestionType) -> None:
        self.type = type

    def __repr__(self) -> str:
        raise NotImplementedError

class TraitQuestion(Question):
    def __init__(self, trait: Trait) -> None:
        """
        Note: We assume that each question is in the affirmative (a.k.a it's asking a yes-question) 
        """
        super().__init__(QuestionType.Trait)
        self.trait = trait
    
    def __repr__(self) -> str:
        if self.trait in [Trait.Eyeglasses, Trait.Blond_hair]:
            return f"Does your individual have {self.trait.value}"
        elif self.trait in [Trait.Male]:
            return f"Is your individual {self.trait.value}"
        else:
            raise Exception(f"[__repr__] Unhandled trait {self.trait}")
        
class GuessQuestion(Question):
    def __init__(self, image_name: str) -> None:
        super().__init__(QuestionType.Guess)
        self.image_name = image_name

    def __repr__(self) -> str:
        return f"Is your person {self.image_name}"