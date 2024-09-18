from enum import Enum
import math
import random

class Trait(Enum):
    Eyeglasses = "Eyeglasses"
    Bald = "Bald"
    Male = "Male"

class Image:
    def __init__(self, name, traits) -> None:
        self.traits: dict[Trait, bool] = traits
        self.name = name
    
    def __repr__(self) -> str:
        return f"Name: {self.name}, Trait: {self.traits}"

class Question:
    def __init__(self, trait: Trait) -> None:
        """
        Note: We assume that each question is in the affirmative (a.k.a it's asking a yes-question) 
        """
        self.trait = trait
    
    def __repr__(self) -> str:
        if self.trait in [Trait.Eyeglasses]:
            return f"Does your individual have {self.trait.value}"
        elif self.trait in [Trait.Bald, Trait.Male]:
            return f"Is your individual {self.trait.value}"

class Solver:
    def compute_question_success_probability(self, images: list[Image], question: Question) -> tuple[float, list[Image], list[Image]]:
        """Returns the probability (between 0 and 1) that the answer to the question True along with the associated images
        """
        if not images:
            return 0, [], []

        i = 0
        yes_images = []
        no_images = []
        for image in images:
            if image.traits.get(question.trait) == True:
                i += 1
                yes_images.append(image)
            else:
                no_images.append(image)
        return i/len(images), yes_images, no_images
        
    def get_best_question(self, images: list[Image], depth: int) -> tuple[Question, float]:
        best_question = None
        highest_expected_entropy = 0
        for trait in Trait:
            question = Question(trait)
            p, yes_imgs, no_imgs = self.compute_question_success_probability(images, question)
            expected_entropy = p * -1 * math.log(p)/math.log(2) + (1 - p) * -1 * math.log(1 - p) / math.log(2) if p not in [0,1] else 0 # expected bits of information from the question (With no lookahead)
            if depth != 0:
                _, expected_entropy_yes = self.get_best_question(yes_imgs, depth-1)
                _, expected_entropy_no = self.get_best_question(no_imgs, depth-1)
                expected_entropy = p * expected_entropy_yes + (1 - p) * expected_entropy_no + expected_entropy

            if expected_entropy >= highest_expected_entropy:
                highest_expected_entropy = expected_entropy
                best_question = question
        return best_question, highest_expected_entropy