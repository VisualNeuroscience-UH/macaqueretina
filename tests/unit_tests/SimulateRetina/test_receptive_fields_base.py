# Built-in
import unittest
from typing import Any, Dict

# Local
from macaqueretina.retina.simulate_retina_module import ReceptiveFieldsBase


class TestReceptiveFieldsBase(unittest.TestCase):
    """
    Unit tests for the ReceptiveFieldsBase class.
    """

    def setUp(self):
        """
        Set up a mock retina dictionary for testing.
        """
        self.mock_retina: Dict[str, Any] = {
            "gc_type": "ON",
            "response_type": "sustained",
            "deg_per_mm": 5.0,
            "dog_model_type": "DoG",
            "spatial_model_type": "gaussian",
            "temporal_model_type": "biphasic",
        }

    def test_initialization(self):
        """
        Test the initialization of ReceptiveFieldsBase.
        """
        rf = ReceptiveFieldsBase(self.mock_retina)

        self.assertEqual(rf.retina_parameters, self.mock_retina)
        self.assertEqual(rf.spatial_filter_sidelen, 0)
        self.assertEqual(rf.microm_per_pix, 0.0)
        self.assertEqual(rf.temporal_filter_len, 0)
        self.assertEqual(rf.gc_type, "ON")
        self.assertEqual(rf.response_type, "sustained")
        self.assertEqual(rf.deg_per_mm, 5.0)
        self.assertEqual(rf.dog_model_type, "DoG")
        self.assertEqual(rf.spatial_model_type, "gaussian")
        self.assertEqual(rf.temporal_model_type, "biphasic")

    def test_missing_retina_parameter(self):
        """
        Test initialization with a missing retina parameter.
        """
        incomplete_retina = self.mock_retina.copy()
        del incomplete_retina["gc_type"]

        with self.assertRaises(KeyError):
            ReceptiveFieldsBase(incomplete_retina)
