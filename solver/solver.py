from enum import Enum
import math
import random
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

class Trait(Enum):
    Eyeglasses = "Eyeglasses"
    Bald = "Bald"
    Male = "Male"

class Answer(Enum):
    """CAREFUL OF CHANGING THIS ORDERING ME USE compute_areas_under_curve() WHICH GOING FROM No TO Yes left-to-right """
    No = 0.1
    Slight_no = 0.3
    Neutral = 0.5
    Slight_yes = 0.7
    Yes = 0.9

class Image:
    def __init__(self, name, traits) -> None:
        self.traits: dict[Trait, float] = traits
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
        else:
            raise Exception(f"[__repr__] Unhandled trait {self.trait}")

class Solver:
    def __init__(self, images: list[Image]) -> None:
        self.images: list[Image] = images

        # Initialize the probs for each trait answer
        self.image_trait_answer_probabilities = {}
        for image in self.images:
            if image.name in self.image_trait_answer_probabilities:
                raise Exception(f"[Solver] Seen too many sussy dupes with name {image.name}")
            self.image_trait_answer_probabilities[image.name] = {}

            for trait in Trait:
                conf = image.traits[trait]
                probs = self.compute_areas_under_curve(conf)
                self.image_trait_answer_probabilities[image.name][trait] = {}
                for i, answer in enumerate(Answer):
                    self.image_trait_answer_probabilities[image.name][trait][answer] = probs[i]

    def compute_areas_under_curve(self, conf: float, std=0.08) -> list[float]:
        """Given a confidence score on a trait, Produce probabilities for each answer under the trait.
        Does this by creating normal distribution and then integrating over each answer region

        Args:
            conf (float): confidence for trait from 0 - 1
            std (float, optional): stand deviation for normal distribution

        Returns:
            list[float]: probabilities for each answer to trait
        """
        # Create the normal distribution
        mean = conf
        distribution = norm(loc=mean, scale=std)

        # Compute the normalization factor (i.e. AOC) for the range from 0 to 1
        normalization_factor, _ = quad(distribution.pdf, 0, 1)
        
        # Normalize the distribution to ensure the area under the curve is 1 in the range [0, 1]
        # Basically just spread evenly until integral of 1 from range [0, 1]
        def normalized_pdf(x):
            return distribution.pdf(x) / normalization_factor

        # Compute the area in each sub-interval
        areas = []
        intervals = []
        i = 0
        while i < 1:
            intervals.append((i,i + 1/len(Answer.__members__)))
            i +=  1/len(Answer.__members__)
        
        for interval in intervals:
            area, _ = quad(normalized_pdf, interval[0], interval[1])
            areas.append(area)
        
        return areas
    
    def solve(self, depth=1) -> tuple[Question, float]:
        probabilities = [1/len(self.images) for i in range(len(self.images))]
        best_expected_entropy = 0
        best_question = None
        for trait in Trait:
            question = Question(trait)
            expected_entropy = 0
            # Compute the expected entropy for said question
            for answer in Answer:
                probability_for_answer = 0
                for i in range(len(self.images)):
                    image = self.images[i]
                    image_p = probabilities[i]
                    p_for_answer = self.image_trait_answer_probabilities[image.name][trait][answer]
                    probability_for_answer += p_for_answer * image_p
                expected_entropy += -1 * probability_for_answer * math.log(probability_for_answer) / math.log(2)    
            # See if best question yet
            if expected_entropy > best_expected_entropy:
                best_expected_entropy = expected_entropy
                best_question = question
            print(f"question {question} as entropy {expected_entropy}")

        if not best_question: raise Exception("[solve] Empty sussy q. Prob a woojwasta responsible.")
        return best_question, best_expected_entropy
