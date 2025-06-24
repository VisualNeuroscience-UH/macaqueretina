# Built-in
import unittest
from unittest.mock import Mock, patch

# Local
from macaqueretina.retina.simulate_retina_module import TemporalModelSubunit


class TestTemporalModelSubunit(unittest.TestCase):

    def setUp(self):
        self.retina_math = Mock()
        self.ganglion_cell = Mock()
        self.response_type = Mock()
        self.cones = Mock()
        self.bipolars = Mock()
        self.model = TemporalModelSubunit(
            self.retina_math,
            self.ganglion_cell,
            self.response_type,
            self.cones,
            self.bipolars,
        )

    def test_initialization(self):
        self.assertIsInstance(self.model, TemporalModelSubunit)
        self.assertEqual(self.model.retina_math, self.retina_math)
        self.assertEqual(self.model.ganglion_cell, self.ganglion_cell)
        self.assertEqual(self.model.response_type, self.response_type)
        self.assertEqual(self.model.cones, self.cones)
        self.assertEqual(self.model.bipolars, self.bipolars)

    def test_create_generator_potential(self):
        vs_mock = Mock()
        gcs_mock = Mock()

        # Mock the _create_dynamic_contrast method
        self.model._create_dynamic_contrast = Mock(return_value=vs_mock)

        # Call the method under test
        result_vs, result_gcs = self.model.create_generator_potential(vs_mock, gcs_mock)

        # Check that methods were called in the correct order with correct arguments
        self.model._create_dynamic_contrast.assert_called_once_with(vs_mock, gcs_mock)
        self.cones.create_signal.assert_called_once_with(vs_mock)
        self.bipolars.create_signal.assert_called_once_with(
            self.cones.create_signal.return_value
        )

        # Check that the results are as expected
        self.assertEqual(result_vs, self.bipolars.create_signal.return_value)
        self.assertEqual(result_gcs, gcs_mock)
