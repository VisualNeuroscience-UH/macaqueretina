# Built-in
from unittest.mock import MagicMock

# Third-party
import pytest

# Local
from macaqueretina.stimuli.experiment_module import RelevantStimulusParameters

# Note that the same class attribute is being tested in a
# different tests.


class TestRelevantStimulusParameters:
    @pytest.fixture
    def mock_options(self):
        return {
            "stimulus_form": "sine_grating",
            "temporal_frequency": 10,
            "spatial_frequency": 20,
            "orientation": 30,
        }

    def test_get_common_relations(self):
        result = RelevantStimulusParameters.get_relations({})
        expected_output = [
            "image_width",
            "image_height",
            "pix_per_deg",
            "fps",
            "duration_seconds",
            "baseline_start_seconds",
            "baseline_end_seconds",
            "pattern",
            "stimulus_form",
            "stimulus_position",
            "stimulus_size",
            "contrast",
            "mean",
            "background",
            "stimulus_video_name",
            "n_sweeps",
            "logarithmic",
        ]
        expected_output.sort()
        result.sort()
        assert result == expected_output

    def test_get_relations_with_real_options(self, mock_options):
        result = RelevantStimulusParameters.get_relations(mock_options)
        expected_output = [
            "image_width",
            "image_height",
            "pix_per_deg",
            "fps",
            "duration_seconds",
            "baseline_start_seconds",
            "baseline_end_seconds",
            "pattern",
            "stimulus_form",
            "stimulus_position",
            "stimulus_size",
            "contrast",
            "mean",
            "background",
            "stimulus_video_name",
            "n_sweeps",
            "logarithmic",
            "temporal_frequency",
            "spatial_frequency",
            "orientation",
        ]
        expected_output.sort()
        result.sort()
        assert result == expected_output
