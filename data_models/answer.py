from enum import Enum

class Answer(Enum):
    """CAREFUL OF CHANGING THIS ORDERING ME USE discretize_confidence_to_probability() WHICH GOING FROM No TO Yes left-to-right """
    No = 0.1
    Slight_no = 0.3
    Neutral = 0.5
    Slight_yes = 0.7
    Yes = 0.9