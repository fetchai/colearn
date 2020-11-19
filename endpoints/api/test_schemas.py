import unittest

import pydantic

from api.schemas import Status, CreateExperiment, ExperimentParameters


class SchemaTests(unittest.TestCase):
    def test_status_state(self):
        with self.assertRaises(pydantic.ValidationError):
            Status(
                experiment='exp',
                state='not-a-valid-state-name',
                epoch=20,
            )

    def test_status_state_valid_status(self):
        for state in ('idle', 'voting', 'training', 'waiting'):
            s = Status(
                experiment='exp',
                state=state,
                epoch=20,
            )
            self.assertIsNotNone(s)

    def test_create_experiment_mode(self):
        with self.assertRaises(pydantic.ValidationError):
            CreateExperiment(
                name='exp-1',
                model='model-1',
                dataset='dataset-1',
                mode='not-a-valid-mode',
                parameters=ExperimentParameters(
                    min_learners=1,
                    max_learners=2,
                    num_epochs=3,
                )
            )

    def test_create_experiment_contract_address_as_follower(self):
        with self.assertRaises(pydantic.ValidationError):
            CreateExperiment(
                name='exp-1',
                model='model-1',
                dataset='dataset-1',
                contract_address=None,
                mode='follower',
                parameters=ExperimentParameters(
                    min_learners=1,
                    max_learners=2,
                    num_epochs=3,
                )
            )

    def test_create_experiment_contract_address_as_owner(self):
        with self.assertRaises(pydantic.ValidationError):
            CreateExperiment(
                name='exp-1',
                model='model-1',
                dataset='dataset-1',
                contract_address='this-value-should-be-none',
                mode='owner',
                parameters=ExperimentParameters(
                    min_learners=1,
                    max_learners=2,
                    num_epochs=3,
                )
            )

