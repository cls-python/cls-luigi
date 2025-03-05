import unittest
from unittest.mock import patch, MagicMock
import warnings
import pandas as pd
import numpy as np
import wandb
from cls_luigi.utils.wandb.core import _log_table


class TestLogTable(unittest.TestCase):
    """Tests for the _log_table function in cls_luigi.utils.wandb.core"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock for wandb.log
        self.wandb_log_patch = patch('wandb.log')
        self.mock_wandb_log = self.wandb_log_patch.start()
        
        # Create a mock for wandb.Table
        self.wandb_table_patch = patch('wandb.Table')
        self.mock_wandb_table = self.wandb_table_patch.start()
        
        # Configure the mock to return itself when called
        self.mock_wandb_table.return_value = MagicMock()
        
        # Create a mock for warnings.warn
        self.warnings_patch = patch('warnings.warn')
        self.mock_warnings = self.warnings_patch.start()

    def tearDown(self):
        """Tear down test fixtures"""
        self.wandb_log_patch.stop()
        self.wandb_table_patch.stop()
        self.warnings_patch.stop()

    def test_existing_wandb_table(self):
        """Test when the value is already a wandb.Table"""
        # Create a mock wandb.Table
        mock_table = MagicMock(spec=wandb.Table)
        
        # Call _log_table with the mock table
        _log_table({"my_table": mock_table})
        
        # Check that wandb.log was called with the correct arguments
        self.mock_wandb_log.assert_called_once_with(
            {"my_table": mock_table}, 
            step=None, 
            commit=True, 
            sync=True
        )
        
        # Verify that wandb.Table was not called (as we already had a Table)
        self.mock_wandb_table.assert_not_called()

    def test_pandas_dataframe(self):
        """Test when the value is a dictionary with a pandas DataFrame"""
        # Create a test DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Call _log_table with the DataFrame
        _log_table({
            "my_table": {
                "data": df,
                "extra_arg": "value"
            }
        })
        
        # Check that wandb.Table was called with the DataFrame
        self.mock_wandb_table.assert_called_once_with(
            dataframe=df,
            extra_arg="value"
        )
        
        # Check that wandb.log was called with the result
        self.mock_wandb_log.assert_called_once()

    def test_dict_with_list_values(self):
        """Test when the value is a dictionary with string keys and list values"""
        # Create a test dictionary
        data_dict = {
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        }
        
        # Call _log_table with the dictionary
        _log_table({
            "my_table": {
                "data": data_dict,
                "extra_arg": "value"
            }
        })
        
        # Check that wandb.Table was called with a DataFrame created from the dict
        self.mock_wandb_table.assert_called_once()
        # Get the first positional argument of the first call
        args, kwargs = self.mock_wandb_table.call_args
        
        # Check that dataframe was passed and extra_arg was passed
        self.assertIn('dataframe', kwargs)
        self.assertEqual(kwargs['extra_arg'], "value")
        
        # Check that wandb.log was called
        self.mock_wandb_log.assert_called_once()

    def test_list_of_lists_with_columns(self):
        """Test when the value is a list of lists with 'columns' in the value"""
        # Create test data
        data = [[1, 'a'], [2, 'b'], [3, 'c']]
        columns = ['col1', 'col2']
        
        # Call _log_table with the list of lists and columns
        _log_table({
            "my_table": {
                "data": data,
                "columns": columns,
                "extra_arg": "value"
            }
        })
        
        # Check that wandb.Table was called with the data and columns
        self.mock_wandb_table.assert_called_once_with(
            columns=columns,
            data=data,
            extra_arg="value"
        )
        
        # Check that wandb.log was called
        self.mock_wandb_log.assert_called_once()

    def test_list_of_dicts(self):
        """Test when the value is a list of dictionaries"""
        # Create test data
        data = [
            {'col1': 1, 'col2': 'a'},
            {'col1': 2, 'col2': 'b'},
            {'col1': 3, 'col2': 'c'}
        ]
        
        # Call _log_table with the list of dictionaries
        _log_table({
            "my_table": {
                "data": data,
                "extra_arg": "value"
            }
        })
        
        # Check that wandb.Table was called with the data
        self.mock_wandb_table.assert_called_once_with(
            data=data,
            extra_arg="value"
        )
        
        # Check that wandb.log was called
        self.mock_wandb_log.assert_called_once()

    def test_numpy_ndarray_with_columns(self):
        """Test when the value is a numpy ndarray with 'columns' in the value"""
        # Create test data
        data = np.array([[1, 2], [3, 4], [5, 6]])
        columns = ['col1', 'col2']
        
        # Call _log_table with the numpy array and columns
        _log_table({
            "my_table": {
                "data": data,
                "columns": columns,
                "extra_arg": "value"
            }
        })
        
        # Check that wandb.Table was called with the data and columns
        self.mock_wandb_table.assert_called_once_with(
            data=data,
            columns=columns,
            extra_arg="value"
        )
        
        # Check that wandb.log was called
        self.mock_wandb_log.assert_called_once()

    def test_dict_with_pandas_series(self):
        """Test when the value is a dictionary with string keys and pandas Series values"""
        # Create test data
        data_dict = {
            'col1': pd.Series([1, 2, 3]),
            'col2': pd.Series(['a', 'b', 'c'])
        }
        
        # Call _log_table with the dictionary of Series
        _log_table({
            "my_table": {
                "data": data_dict,
                "extra_arg": "value"
            }
        })
        
        # Check that wandb.Table was called with a DataFrame created from the dict
        self.mock_wandb_table.assert_called_once()
        # Get the first positional argument of the first call
        args, kwargs = self.mock_wandb_table.call_args
        
        # Check that dataframe was passed and extra_arg was passed
        self.assertIn('dataframe', kwargs)
        self.assertEqual(kwargs['extra_arg'], "value")
        
        # Check that wandb.log was called
        self.mock_wandb_log.assert_called_once()

    def test_unsupported_table_format(self):
        """Test when the value has an unsupported table format"""
        # Call _log_table with an unsupported format
        _log_table({
            "my_table": {
                "data": 123,  # Not a supported format
                "extra_arg": "value"
            }
        })
        
        # Check that a warning was issued
        self.mock_warnings.assert_called_once()
        
        # Check that wandb.Table was not called
        self.mock_wandb_table.assert_not_called()
        
        # Check that wandb.log was not called (no processed data)
        self.mock_wandb_log.assert_not_called()

    def test_multiple_tables(self):
        """Test when multiple tables are provided"""
        # Create test data
        df1 = pd.DataFrame({'col1': [1, 2, 3]})
        df2 = pd.DataFrame({'col2': [4, 5, 6]})
        
        # Configure the mock to return different values for each call
        mock_table1 = MagicMock()
        mock_table2 = MagicMock()
        self.mock_wandb_table.side_effect = [mock_table1, mock_table2]
        
        # Call _log_table with multiple tables
        _log_table({
            "table1": {"data": df1},
            "table2": {"data": df2}
        })
        
        # Check that wandb.Table was called twice
        self.assertEqual(self.mock_wandb_table.call_count, 2)
        
        # Check that wandb.log was called with both tables
        expected_log = {
            "table1": mock_table1,
            "table2": mock_table2
        }
        self.mock_wandb_log.assert_called_once_with(
            expected_log,
            step=None,
            commit=True,
            sync=True
        )

    def test_custom_step_commit_sync(self):
        """Test with custom step, commit, and sync values"""
        # Create a test DataFrame
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Call _log_table with custom parameters
        _log_table(
            {"my_table": {"data": df}},
            step=42,
            commit=False,
            sync=False
        )
        
        # Check that wandb.log was called with the custom parameters
        self.mock_wandb_log.assert_called_once_with(
            {"my_table": self.mock_wandb_table.return_value},
            step=42,
            commit=False,
            sync=False
        )

    def test_empty_input(self):
        """Test with an empty input dictionary"""
        # Call _log_table with an empty dictionary
        _log_table({})
        
        # Check that wandb.log was not called
        self.mock_wandb_log.assert_not_called()


if __name__ == '__main__':
    unittest.main()
