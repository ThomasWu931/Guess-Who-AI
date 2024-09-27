from enum import Enum
import math
import random
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

"""
TODO:

1. Add support to dynamically update the image probabilities as the game goes one
2. Add beyond depth 1 searches
"""

class Trait(Enum):
    Eyeglasses = "Eyeglasses"
    # Bald = "Bald"
    Male = "Male"
    Blond_hair = "Blond_hair"

class Answer(Enum):
    """CAREFUL OF CHANGING THIS ORDERING ME USE discretize_confidence_to_probability() WHICH GOING FROM No TO Yes left-to-right """
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

class TraitQuestion:
    def __init__(self, trait: Trait) -> None:
        """
        Note: We assume that each question is in the affirmative (a.k.a it's asking a yes-question) 
        """
        self.trait = trait
    
    def __repr__(self) -> str:
        if self.trait in [Trait.Eyeglasses, Trait.Blond_hair]:
            return f"Does your individual have {self.trait.value}"
        elif self.trait in [Trait.Male]:
            return f"Is your individual {self.trait.value}"
        else:
            raise Exception(f"[__repr__] Unhandled trait {self.trait}")

class GuessQuestion:
    def __init__(self, image_name: str) -> None:
        self.image_name = image_name

    def __repr__(self) -> str:
        return f"Is your person {self.image_name}"

class Solver:
    def __init__(self, images: list[Image], verbose=False) -> None:
        self.questions: list[TraitQuestion | GuessQuestion] = []
        self.answers: list[Answer] = []
        self.images: list[Image] = images
        self.verbose = verbose
        self.image_trait_answer_probabilities = self._compute_image_probabilities_from_confidence(self.images)
        self.image_probabilities: dict[str, float] = {image.name: 1/len(self.images) for image in self.images}
    
    def print(self, *args, **xargs):
        if self.verbose:
            print(*args, **xargs)

    def _compute_image_probabilities_from_confidence(self, images: list[Image]) -> dict[str, dict[Trait, dict[Answer, float]]]:
        """Given images, convert all of their confidence scores into conditional probabilities P(a | x ^ y) where a is trait, x is
        some image that could be chosen, and y is the answer given for said trait

        Args:
            images (list[Image]): images to be processed

        Returns:
            dict[str, dict[Trait, dict[Answer, float]]]: image name -> trait -> answer -> conditional probability
        """
        # Initialize the probs for each trait answer
        image_trait_answer_probabilities = {}
        for image in images:
            if image.name in image_trait_answer_probabilities:
                raise Exception(f"[Solver] Seen too many sussy dupes with name {image.name}")
            image_trait_answer_probabilities[image.name] = {}

            for trait in Trait:
                conf = image.traits[trait]
                probs = self._discretize_confidence_to_probability(conf)
                image_trait_answer_probabilities[image.name][trait] = {}
                for i, answer in enumerate(Answer):
                    image_trait_answer_probabilities[image.name][trait][answer] = probs[i]
        self.print(f"[image_trait_answer_probabilities]: {image_trait_answer_probabilities}", end='\n\n')
        return image_trait_answer_probabilities

    def _discretize_confidence_to_probability(self, conf: float, std=0.08) -> list[float]:
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
        
        if abs(sum(areas) - 1) >= 0.01:
            raise Exception(f"[_discretize_confidence_to_probability] {areas}")
        return areas
    
    def _get_p_image_given_questions_and_answers(self, image: Image, questions: list[TraitQuestion], answers: list[Answer]) -> dict[str, float]:
        """Given the current questions and answers, compute the probability for each image

        Notes:
            - We assume that all traits are conditionally independent under the assumption that the chosen image x
            (i.e. we assume Naïve Bayes)
            - https://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf for formula

        Returns:
            dict[str, float]: maps image names to their probability
        """
        other_images = list(filter(lambda x: x != image, self.images))
        p_trait_given_image = 1/len(self.images)
        p_trait_given_not_image = [1/len(self.images) for i in range(len(other_images))]

        for q, a in zip(questions, answers):
            # Compute p_trait_given_image
            p_trait_given_image *= self.image_trait_answer_probabilities[image.name][q.trait][a]

            # Compute p_trait_given_not_image
            for i in range(len(other_images)):
                other_image = other_images[i]
                p_trait_given_not_image[i] *= self.image_trait_answer_probabilities[other_image.name][q.trait][a]

        return p_trait_given_image/(p_trait_given_image+sum(p_trait_given_not_image))

    def process_question_and_answer(self, question: TraitQuestion | GuessQuestion, answer: Answer) -> None:
        """Given a new question and answer, update the conditional probabilities for each image

        Args:
            question (Question): q
            answer (Answer): a
        """
        if type(question) == GuessQuestion:
            # When we have a guess question, we just delete the image from the the images array
            image_name = question.image_name
            self.images = list(filter(lambda x: x.name != image_name, self.images))
            del self.image_trait_answer_probabilities[image_name]

        elif type(question) == TraitQuestion:
            self.questions.append(question)
            self.answers.append(answer)

        # Recompute image probabilities
        for image in self.images:
            p = self._get_p_image_given_questions_and_answers(image, self.questions, self.answers)
            self.image_probabilities[image.name] = p

        self.print(f"\n[image_probabilities]: {self.image_probabilities}",end='\n\n')

        if abs(sum(self.image_probabilities.values()) - 1) >= 0.01:
            raise Exception(f"[process_question_and_answer] Leaking woojer sum {sum(self.image_probabilities.values())}. {self.image_probabilities.values()}")

    def get_best_question(self, depth=1) -> GuessQuestion | TraitQuestion:
        best_expected_entropy = 0
        best_question = None
        for trait in Trait:
            question = TraitQuestion(trait)
            expected_entropy = 0
            # Compute the expected entropy for said question
            for answer in Answer:
                probability_for_answer = 0
                for i in range(len(self.images)):
                    image = self.images[i]
                    image_p = self.image_probabilities[image.name]
                    p_for_answer = self.image_trait_answer_probabilities[image.name][trait][answer]
                    probability_for_answer += p_for_answer * image_p

                if probability_for_answer != 0: # In case it's 0, then we gain no info so expected_entropy remains the same
                    expected_entropy += -1 * probability_for_answer * math.log(probability_for_answer) / math.log(2)    
            # See if best question yet
            if expected_entropy > best_expected_entropy:
                best_expected_entropy = expected_entropy
                best_question = question
            self.print(f"Question: {question} as entropy {expected_entropy}")

        if not best_question: raise Exception("[solve] Empty sussy q. Prob a woojwasta responsible.")

        # Consider whether to ask a trait or guess question
        # Use use the heuristic that if less than 80% of the data is expected to be cleared, then, we'd use a question over a final guess
        if best_expected_entropy > 0.32:
            self.print(f"BEST QUESTION: {best_question} produced expected entropy {best_expected_entropy}")
        else:
            image_choice = random.choices(self.images, weights=self.image_probabilities, k=1)[0]
            best_question = GuessQuestion(image_choice)
        
        return best_question    



# class Solver:
#     def compute_question_success_probability(self, images: list[Image], question: Question) -> tuple[float, list[Image], list[Image]]:
#         """Returns the probability (between 0 and 1) that the answer to the question True along with the associated images
#         """
#         if not images:
#             return 0, [], []

#         i = 0
#         yes_images = []
#         no_images = []
#         for image in images:
#             if image.traits.get(question.trait) == True:
#                 i += 1
#                 yes_images.append(image)
#             else:
#                 no_images.append(image)
#         return i/len(images), yes_images, no_images
        
#     def get_best_question(self, images: list[Image], depth: int) -> tuple[Question, float]:
#         best_question = None
#         highest_expected_entropy = 0
#         for trait in Trait:
#             question = Question(trait)
#             p, yes_imgs, no_imgs = self.compute_question_success_probability(images, question)
#             expected_entropy = p * -1 * math.log(p)/math.log(2) + (1 - p) * -1 * math.log(1 - p) / math.log(2) if p not in [0,1] else 0 # expected bits of information from the question (With no lookahead)
#             if depth != 0:
#                 _, expected_entropy_yes = self.get_best_question(yes_imgs, depth-1)
#                 _, expected_entropy_no = self.get_best_question(no_imgs, depth-1)
#                 expected_entropy = p * expected_entropy_yes + (1 - p) * expected_entropy_no + expected_entropy

#             if expected_entropy >= highest_expected_entropy:
#                 highest_expected_entropy = expected_entropy
#                 best_question = question
#         return best_question, highest_expected_entropy
    

    # def get_image_probabilities(self, questions: list[Question], answers: list[Answer]):
    #     probabilities = [-1 for i in range(len(self.images))]

    #     numerator = 1
    #     for i in range(len(probabilities)):
    #         image = self.images[i]
    #         numerator = 1
    #         denominator = 1
    #         p_trait_given_answer = 1/len(self.images)
    #         p_trait_given_not_answer = 1 - 1/len(self.images)
    #         for q, a in zip(questions, answers):
    #             trait = q.trait
    #             p_trait_given_answer *= max(0.005, self.compute_prob_trait_given_image(image.traits[trait], a.value)) # Probability that a character's trait is true given an answer
    #             p_trait_given_not_answer *= max(0.005,sum([self.compute_prob_trait_given_image(other_img.traits[trait], a.value) if image != other_img else 0 for other_img in self.images])/(len(self.images) - 1))

    #         probabilities[i] = numerator/(numerator+denominator)

    #     # Normalize the probabilities
    #     if abs(sum(probabilities) - 1) >= 0.01:
    #         raise Exception("ERORR")
    #     probabilities = [p for p in probabilities]

    #     return probabilities
    