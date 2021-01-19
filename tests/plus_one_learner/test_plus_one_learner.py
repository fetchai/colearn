from tests.plus_one_learner.plus_one_learner import PlusOneLearner


def test_init():
    learner = PlusOneLearner(0)
    assert learner.current_value == 0
    learner_2 = PlusOneLearner(2)
    assert learner_2.current_value == 2
