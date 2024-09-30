from enum import Enum
import math
import random
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from ..data_models import *

"""
TODO:

1. Add support to dynamically update the image probabilities as the game goes one
2. Add beyond depth 1 searches
"""

class Solver:
    def __init__(self, images: list[Image], verbose=False) -> None:
        """
        Notes:
            - Even though we take in Question[], we ultimately remove the GuessQuestion, using them to eliminate choices
        """
        self.questions: list[TraitQuestion] = []
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

    def process_question_and_answer(self, question: Question, answer: Answer) -> None:
        """Given a new question and answer, update the conditional probabilities for each image

        Args:
            question (Question): q
            answer (Answer): a
        """
        if question.type == QuestionType.Guess:
            # When we have a guess question, we just delete the image from the the images array
            old_length = len(self.images)
            image_name = question.image_name
            self.images = list(filter(lambda x: x.name != image_name, self.images))
            assert old_length - 1 == len(self.images)
            del self.image_trait_answer_probabilities[image_name]
            del self.image_probabilities[image_name]
        elif question.type == QuestionType.Trait:
            self.questions.append(question)
            self.answers.append(answer)
        else:
            raise Exception(f"[process_question_and_answer] Invalid question type {question.type}")

        # Recompute image probabilities
        for image in self.images:
            p = self._get_p_image_given_questions_and_answers(image, self.questions, self.answers)
            self.image_probabilities[image.name] = p

        self.print(f"\n[image_probabilities]: {[round(p, 2) for p in self.image_probabilities.values()]}",end='\n\n')

        if abs(sum(self.image_probabilities.values()) - 1) >= 0.01:
            raise Exception(f"[process_question_and_answer] Leaking woojer sum {sum(self.image_probabilities.values())}. {list(self.image_probabilities.values())}")

    def _compute_entropy_of_list(self, li: list[float]):
        entropy = 0
        for p in li:
            if p > 0:  # To avoid log(0) which is undefined
                entropy -= p * math.log2(p)
        return entropy

    def get_best_question(self, depth=1) -> TraitQuestion | GuessQuestion:
        best_expected_entropy = 10 ** 10
        best_trait_question = None

        # Consider getting the best trait question
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

                # Compute the entrpy of the image probs if this question is answered
                future_questions, future_answers = self.questions + [question], self.answers + [answer]
                future_probs = [self._get_p_image_given_questions_and_answers(image, future_questions, future_answers) for image in self.images]
                future_entropy = self._compute_entropy_of_list(future_probs) # TODO: clarify that we aren't really computing expected entropy

                expected_entropy += probability_for_answer * future_entropy # TODO: clarify that we aren't really computing expected entropy and more like an expected future entropy
            # See if best question yet
            if expected_entropy < best_expected_entropy:
                best_expected_entropy = expected_entropy
                best_trait_question = question
            self.print(f"Question: {question} as entropy {expected_entropy}")
        if not best_trait_question: raise Exception("[solve] Empty sussy t q. Prob a woojwasta responsible.")
        self.print(f"Current entropy: {round(current_entropy, 2)}. Future entropy: {round(best_expected_entropy, 2)}. Best trait question: {repr(best_trait_question)}")

        """
        Consider whether to ask a trait or guess question
        We consider two conditions
            #1: Has this question been asked in the past? 
            #2: Do we decrease entropy enough by asking the question? 
        """


        best_question = None
        if best_trait_question in self.questions or abs(expected_entropy - current_entropy) < 0.2 or current_entropy < 0.1:
            image_choice = random.choices(self.images, weights=[self.image_probabilities[img.name] for img in self.images], k=1)[0]
            best_question = GuessQuestion(image_choice.name)
        else:
            best_question = best_trait_question
        self.print(f"BEST QUESTION: {best_question}")

        current_entropy = self._compute_entropy_of_list(self.image_probabilities.values())
        return best_question  
