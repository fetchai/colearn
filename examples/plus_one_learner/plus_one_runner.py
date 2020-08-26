from colearn_interface.standalone_driver import run

from examples.plus_one_learner.plus_one_learner import PlusOneLearner


def run_experiment(n_learners):
    learners = [PlusOneLearner(0)] * 5

    for l in learners:
        print(l.current_value)

    run(10, learners)

    for l in learners:
        print(l.current_value)


if __name__ == "__main__":
    # execute only if run as a script
    run_experiment(5)
