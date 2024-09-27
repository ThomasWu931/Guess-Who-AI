from ..solver.solver import *

# # Hardcoded sample images for testing
# image1 = Image(
#     "Image_1",
#     {
#         Trait.Eyeglasses: 0.9,  # High confidence this individual has eyeglasses
#         Trait.Bald: 0.1,        # Low confidence this individual is bald
#         Trait.Male: 0.8         # High confidence this individual is male
#     }
# )

# image2 = Image(
#     "Image_2",
#     {
#         Trait.Eyeglasses: 0.2,  # Low confidence this individual has eyeglasses
#         Trait.Bald: 0.7,        # High confidence this individual is bald
#         Trait.Male: 0.5         # Neutral confidence this individual is male
#     }
# )

# image3 = Image(
#     "Image_3",
#     {
#         Trait.Eyeglasses: 0.4,  # Medium confidence this individual has eyeglasses
#         Trait.Bald: 0.6,        # Medium confidence this individual is bald
#         Trait.Male: 0.3         # Low confidence this individual is male
#     }
# )

# image4 = Image(
#     "Image_4",
#     {
#         Trait.Eyeglasses: 0.8,  # High confidence this individual has eyeglasses
#         Trait.Bald: 0.2,        # Low confidence this individual is bald
#         Trait.Male: 0.6         # Medium confidence this individual is male
#     }
# )

# image5 = Image(
#     "Image_5",
#     {
#         Trait.Eyeglasses: 0.3,  # Low confidence this individual has eyeglasses
#         Trait.Bald: 0.9,        # High confidence this individual is bald
#         Trait.Male: 0.4         # Medium confidence this individual is male
#     }
# )

# # Hardcoded list of images
# hardcoded_images = [image1, image2, image3, image4, image5]

# # Print out the hardcoded images
# for img in hardcoded_images:
#     print(img)

# # Initialize the solver with the hardcoded images
# solver = Solver(hardcoded_images)

# # Test the solver to get the best question and expected entropy
# best_question, expected_entropy = solver.solve()

# # Output the results
# print(f"Best Question: {best_question}")
# print(f"Expected Entropy: {expected_entropy}")


def test_no_dumb_choice():
    # Should choose male question since it has the most uniform distribution for each answer
    images = [
        Image(
            "Image_1",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Bald: 0,        # High confidence this individual is bald
                Trait.Male: 0.1        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_2",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Bald: 0,        # High confidence this individual is bald
                Trait.Male: 0.3        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_3",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Bald: 1,        # High confidence this individual is bald
                Trait.Male: 0.5        # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_4",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Bald: 0.9,        # High confidence this individual is bald
                Trait.Male: 0.7         # Medium confidence this individual is male
            }
        ),
        Image(
            "Image_5",
            {
                Trait.Eyeglasses: 1,  # Low confidence this individual has eyeglasses
                Trait.Bald: 0.8,        # High confidence this individual is bald
                Trait.Male: 0.9         # Medium confidence this individual is male
            }
        )
    ]
    s = Solver(images)
    q, e = s.solve()
    print(q, e)


test_no_dumb_choice()