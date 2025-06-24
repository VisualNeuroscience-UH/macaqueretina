# Built-in
import unittest
from typing import Any, List
from unittest.mock import Mock

# Local
# Assuming SimulationDirector is in a file named simulation_director.py
from macaqueretina.retina.simulate_retina_module import SimulationDirector


class TestSimulationDirector(unittest.TestCase):
    """
    Unit tests for the SimulationDirector class.
    """

    def setUp(self):
        """
        Set up a mock builder and SimulationDirector instance for testing.
        """
        self.mock_builder = Mock()
        self.director = SimulationDirector(self.mock_builder)

    def test_initialization(self):
        """
        Test the initialization of SimulationDirector.
        """
        self.assertIsInstance(self.director, SimulationDirector)
        self.assertEqual(self.director.builder, self.mock_builder)

    def test_run_simulation(self):
        """
        Test the run_simulation method.
        """
        self.director.run_simulation()

        # Verify that all expected methods of the builder were called
        self.mock_builder.get_concrete_components.assert_called_once()
        self.mock_builder.create_spatial_filters.assert_called_once()
        self.mock_builder.get_spatially_cropped_video.assert_called_once()
        self.mock_builder.get_noise.assert_called_once()
        self.mock_builder.get_generator_potentials.assert_called_once()
        self.mock_builder.generate_spikes.assert_called_once()

    def test_run_impulse_response(self):
        """
        Test the run_impulse_response method.
        """
        contrasts: List[float] = [0.1, 0.2, 0.3]
        self.director.run_impulse_response(contrasts)

        self.mock_builder.get_concrete_components.assert_called_once()
        self.mock_builder.get_impulse_response.assert_called_once_with(contrasts)

    def test_run_uniformity_index(self):
        """
        Test the run_uniformity_index method.
        """
        self.director.run_uniformity_index()

        self.mock_builder.get_concrete_components.assert_called_once()
        self.mock_builder.create_spatial_filters.assert_called_once()
        self.mock_builder.get_uniformity_index.assert_called_once()

    def test_get_simulation_result(self):
        """
        Test the get_simulation_result method.
        """
        # Mock the vs and gcs attributes of the builder
        mock_vs = Mock()
        mock_gcs = Mock()
        self.mock_builder.vs = mock_vs
        self.mock_builder.gcs = mock_gcs

        result = self.director.get_simulation_result()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], mock_vs)
        self.assertEqual(result[1], mock_gcs)
