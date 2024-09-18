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


def generate_random_images(num_samples=100):
    images = []
    for i in range(1, num_samples + 1):
        # Randomly assign True/False for each trait
        traits = {
            Trait.Eyeglasses: random.choice([True, False]),
            Trait.Bald: random.choice([True, False]),
            Trait.Male: random.choice([True, False])
        }
        images.append(Image(f"Person{i}", traits))
    return images

def get_random_question(images):
    random_trait = random.choice(list(Trait))
    random_value = random.choice([True, False])
    question = Question(random_trait, random_value)
    if len(images) <= 10:
        entropy = 0
    else:
        entropy = 1 
    return question, entropy

def solve():
    s = Solver()
    avg_time = None
    max_depth = 3
    avg_e = [None for i in range(max_depth)]
    for i in range(1, 100 + 1):
        for depth in range(max_depth):
            images: list[Image] = generate_random_images(20000)
            _, entropy = s.get_best_question(images, depth)
            e = avg_e[depth]
            if not e:
                e = entropy
            else:
                e = (entropy + e * i) / (i + 1)
            avg_e[depth] = e
        print("entropy", sorted([(e / (k + 1), k) for k, e in enumerate(avg_e)]), len(set(avg_e)))
        # time = 0
        # while len(images) > 1:
        #     question, entropy = s.get_best_question(images, 0)
        #     # question, entropy = get_random_question(images)
        #     time += 1
        #     if entropy == 0:
        #         breakpoint()
        #         images.remove(random.choice(images))
        #     else:
        #         # Try to answer the question
        #         newImages = []
        #         for image in images:
        #             if image.traits.get(question.trait) == question.v:
        #                 newImages.append(image)
        #         images = newImages
        # if not avg_time:
        #     avg_time = time
        # else:
        #     avg_time = (time + avg_time * i) / (i + 1)
        # print(avg_time)

#solve()

s = Solver() 
images = [
    Image("Person2", {Trait.Eyeglasses: False, Trait.Bald: True}),
    Image("Person3", {Trait.Eyeglasses: False, Trait.Bald: False}),
    Image("Person4", {Trait.Eyeglasses: False, Trait.Bald: True}),
    Image("Person5", {Trait.Eyeglasses: True, Trait.Bald: False}),
    Image("Person5", {Trait.Eyeglasses: True, Trait.Bald: True}),
    Image("Person5", {Trait.Eyeglasses: True, Trait.Bald: True})
]
q, entropy = s.get_best_question(images, 1)
print(q, entropy)