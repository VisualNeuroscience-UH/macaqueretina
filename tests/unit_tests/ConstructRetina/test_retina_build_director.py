# Built-in
from unittest.mock import MagicMock

# Local
from macaqueretina.retina.construct_retina_module import RetinaBuildDirector


class TestRetinaBuildDirector:
    def setup_method(self):
        # Create a mock builder object
        self.builder_mock = MagicMock()

        # Initialize the RetinaBuildDirector with the mock builder
        self.director = RetinaBuildDirector(builder=self.builder_mock)

    def test_construct_retina(self):
        """
        Test the construct_retina method to ensure it calls the right methods in the correct order.
        """
        # Call the construct_retina method
        self.director.construct_retina()

        # Verify that each of the builder's methods was called exactly once
        self.builder_mock.get_concrete_components.assert_called_once()
        self.builder_mock.fit_cell_density_data.assert_called_once()
        self.builder_mock.place_units.assert_called_once()
        self.builder_mock.create_spatial_receptive_fields.assert_called_once()
        self.builder_mock.connect_units.assert_called_once()
        self.builder_mock.create_temporal_receptive_fields.assert_called_once()
        self.builder_mock.create_tonic_drive.assert_called_once()

    def test_get_retina(self):
        """
        Test the get_retina method to ensure it returns the builder's retina and ganglion_cell.
        """
        # Set mock return values for retina and ganglion_cell
        mock_retina = MagicMock()
        mock_ganglion_cell = MagicMock()
        self.builder_mock.retina = mock_retina
        self.builder_mock.ganglion_cell = mock_ganglion_cell

        # Call the get_retina method
        retina, ganglion_cell = self.director.get_retina()

        # Verify that the returned retina and ganglion_cell match the mock values
        assert retina == mock_retina
        assert ganglion_cell == mock_ganglion_cell
