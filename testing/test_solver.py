from ..solver.solver import *
from ..data_models import *

def basic_trait_question_test():
    images = [
        Image(
            "Image_1",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 0,        # High confidence this individual is bald
                Trait.Male: 0.1        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_2",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 0.5,        # High confidence this individual is bald
                Trait.Male: 0.3        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_3",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 1,        # High confidence this individual is bald
                Trait.Male: 0.5        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_4",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 0.9,        # High confidence this individual is bald
                Trait.Male: 0.7         # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_5",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 0.8,        # High confidence this individual is bald
                Trait.Male: 0.9         # Medium confidence this individual is male
            }
        )
    ]
    s = Solver(images,verbose=True)
    q = s.get_best_question()
    # Should choose male question since it has the most uniform distribution for each answer
    assert repr(q) == "Is your individual Male"

    a = Answer.Yes
    s.process_question_and_answer(q, a)

    q = s.get_best_question()
    assert repr(q) == "Does your individual have Blond_hair"

    a = Answer.No
    s.process_question_and_answer(q, a)

def basic_guess_question_test():
    random.seed(1)
    images = [
        Image(
            "Image_1",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 1,        # High confidence this individual is bald
                Trait.Male: 1        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_2",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 1,        # High confidence this individual is bald
                Trait.Male: 1        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_3",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 1,        # High confidence this individual is bald
                Trait.Male: 1        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_4",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 1,        # High confidence this individual is bald
                Trait.Male: 1         # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_5",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 0.9,        # High confidence this individual is bald
                Trait.Male: 1         # Medium confidence this individual is male
            }
        )
    ]
    s = Solver(images,verbose=True)
    q = s.get_best_question()
    assert repr(q) == "Is your person Image_1"

def basic_guess_question_test_2():
    random.seed(2)
    images = [
        Image(
            "Image_1",
            {
                Trait.Eyeglasses: 0,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 1,        # High confidence this individual is bald
                Trait.Male: 1        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_2",
            {
                Trait.Eyeglasses: 0,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 1,        # High confidence this individual is bald
                Trait.Male: 0        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_3",
            {
                Trait.Eyeglasses: 0,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 0,        # High confidence this individual is bald
                Trait.Male: 0        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_4",
            {
                Trait.Eyeglasses: 0,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 0,        # High confidence this individual is bald
                Trait.Male: 0         # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_5",
            {
                Trait.Eyeglasses: 0,  # Low confidence this individual has eyeglasses
                Trait.Blond_hair: 0,        # High confidence this individual is bald
                Trait.Male: 0         # Medium confidence this individual is male
            }
        )
    ]
    s = Solver(images,verbose=True)
    q = s.get_best_question()
    s.process_question_and_answer(q, Answer.Yes)
    q = s.get_best_question()
    s.process_question_and_answer(q, Answer.Yes)
    q = s.get_best_question()
    assert repr(q) == "Is your person Image_1"
    s.process_question_and_answer(q, Answer.No)
    breakpoint()

# basic_trait_question_test()
basic_guess_question_test_2()

