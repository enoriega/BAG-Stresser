import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from stresser import (
    simulate_user_session,
    get_available_models,
    print_session_report,
    UserSessionStats,
    ConversationStats
)


class TestGetAvailableModels:
    """Tests for get_available_models function."""

    @patch('openai.OpenAI')
    def test_successful_model_fetch(self, mock_openai_class):
        """Test successful fetching of models from API."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_model1 = Mock()
        mock_model1.id = 'model-1'
        mock_model2 = Mock()
        mock_model2.id = 'model-2'

        mock_models = Mock()
        mock_models.data = [mock_model1, mock_model2]
        mock_client.models.list.return_value = mock_models

        # Test
        models = get_available_models('test_key', 'https://test.api')

        assert models == ['model-1', 'model-2']
        mock_openai_class.assert_called_once_with(
            api_key='test_key',
            base_url='https://test.api'
        )

    @patch('openai.OpenAI')
    @patch('builtins.print')
    def test_fallback_on_error(self, mock_print, mock_openai_class):
        """Test fallback to default models when API call fails."""
        mock_openai_class.side_effect = Exception("API Error")

        models = get_available_models('test_key', 'https://test.api')

        assert models == ['gpt-3.5-turbo', 'gpt-4']
        # Verify warning was printed
        mock_print.assert_called_once()
        assert 'Warning' in str(mock_print.call_args)


class TestSimulateUserSession:
    """Tests for simulate_user_session function."""

    @pytest.mark.asyncio
    async def test_missing_api_credentials(self):
        """Test that ValueError is raised when credentials are missing."""
        with pytest.raises(ValueError, match="api_key and api_base are required"):
            await simulate_user_session(api_key=None, api_base='https://test.api')

        with pytest.raises(ValueError, match="api_key and api_base are required"):
            await simulate_user_session(api_key='key', api_base=None)

    @pytest.mark.asyncio
    async def test_nonexistent_conversations_directory(self):
        """Test that FileNotFoundError is raised for missing directory."""
        with pytest.raises(FileNotFoundError, match="Conversations directory not found"):
            await simulate_user_session(
                conversations_dir='/nonexistent/path',
                api_key='key',
                api_base='https://test.api'
            )

    @pytest.mark.asyncio
    async def test_empty_conversations_directory(self, tmp_path):
        """Test that FileNotFoundError is raised for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No conversation files found"):
            await simulate_user_session(
                conversations_dir=str(empty_dir),
                api_key='key',
                api_base='https://test.api'
            )

    @pytest.mark.asyncio
    @patch('stresser.get_available_models')
    @patch('stresser.run_conversation_stress_test', new_callable=AsyncMock)
    @patch('stresser.time.time')
    @patch('builtins.print')
    async def test_short_duration_session(
        self,
        mock_print,
        mock_time,
        mock_run_test,
        mock_get_models,
        tmp_path
    ):
        """Test a short session with fixed model."""
        # Create test conversation file
        conv_file = tmp_path / "test_conv.json"
        conv_file.write_text('{"conversation_id": "test", "messages": []}')

        # Mock time to control loop duration (run 2 iterations)
        start_time = 1000.0
        mock_time.side_effect = [
            start_time,  # Initial time
            start_time + 0.5,  # First iteration check
            start_time + 1.5,  # Second iteration check
            start_time + 2.5,  # Third iteration check (exit)
        ]

        # Mock successful conversation runs
        mock_stats = ConversationStats(conversation_id='test')
        mock_stats.total_messages_sent = 5
        mock_stats.total_tokens = 100
        mock_stats.total_tokens_input = 50
        mock_stats.total_tokens_output = 50
        mock_stats.average_latency_seconds = 1.5
        mock_stats.message_latencies = [1.0, 1.5, 2.0]
        mock_run_test.return_value = mock_stats

        # Run session with fixed model (duration=2 seconds)
        result = await simulate_user_session(
            conversations_dir=str(tmp_path),
            duration_seconds=2,
            model_name='test-model',
            api_key='test_key',
            api_base='https://test.api'
        )

        # Verify results
        assert result.total_conversations == 2
        assert result.successful_conversations == 2
        assert result.failed_conversations == 0
        assert result.total_messages_sent == 10  # 5 * 2
        assert result.total_tokens == 200  # 100 * 2
        assert result.models_used == ['test-model']
        assert mock_run_test.call_count == 2
        # Should not call get_available_models when model is specified
        mock_get_models.assert_not_called()

    @pytest.mark.asyncio
    @patch('stresser.get_available_models')
    @patch('stresser.run_conversation_stress_test', new_callable=AsyncMock)
    @patch('stresser.time.time')
    @patch('builtins.print')
    async def test_random_model_selection(
        self,
        mock_print,
        mock_time,
        mock_run_test,
        mock_get_models,
        tmp_path
    ):
        """Test session with random model selection."""
        # Create test conversation file
        conv_file = tmp_path / "test_conv.json"
        conv_file.write_text('{"conversation_id": "test", "messages": []}')

        # Mock available models
        mock_get_models.return_value = ['model-a', 'model-b']

        # Mock time (1 iteration)
        start_time = 1000.0
        mock_time.side_effect = [
            start_time,
            start_time + 0.5,
            start_time + 2.0,  # Exit
        ]

        # Mock successful run
        mock_stats = ConversationStats(conversation_id='test')
        mock_stats.total_messages_sent = 3
        mock_stats.total_tokens = 50
        mock_stats.total_tokens_input = 25
        mock_stats.total_tokens_output = 25
        mock_stats.message_latencies = [1.0]
        mock_run_test.return_value = mock_stats

        # Run without specifying model
        result = await simulate_user_session(
            conversations_dir=str(tmp_path),
            duration_seconds=1,
            model_name=None,  # Random selection
            api_key='test_key',
            api_base='https://test.api'
        )

        # Verify get_available_models was called
        mock_get_models.assert_called_once_with('test_key', 'https://test.api')
        # Verify one of the models was used
        assert len(result.models_used) == 1
        assert result.models_used[0] in ['model-a', 'model-b']

    @pytest.mark.asyncio
    @patch('stresser.get_available_models')
    @patch('stresser.run_conversation_stress_test', new_callable=AsyncMock)
    @patch('stresser.time.time')
    @patch('builtins.print')
    async def test_error_handling(
        self,
        mock_print,
        mock_time,
        mock_run_test,
        mock_get_models,
        tmp_path
    ):
        """Test that errors are properly tracked."""
        # Create test conversation file
        conv_file = tmp_path / "test_conv.json"
        conv_file.write_text('{"conversation_id": "test", "messages": []}')

        # Mock time (1 iteration - will only process one before timeout)
        start_time = 1000.0
        mock_time.side_effect = [
            start_time,  # Initial start
            start_time + 0.5,  # First iteration check
            start_time + 2.0,  # Exit (exceeded duration)
        ]

        # Only one call - fails
        fail_stats = ConversationStats(conversation_id='test')
        fail_stats.error = "API Error"

        mock_run_test.side_effect = [fail_stats]

        # Run session
        result = await simulate_user_session(
            conversations_dir=str(tmp_path),
            duration_seconds=1,
            model_name='test-model',
            api_key='test_key',
            api_base='https://test.api'
        )

        # Verify error tracking
        assert result.total_conversations == 1
        assert result.successful_conversations == 0
        assert result.failed_conversations == 1
        assert result.total_messages_sent == 0  # None successful

    @pytest.mark.asyncio
    @patch('stresser.get_available_models')
    @patch('stresser.run_conversation_stress_test', new_callable=AsyncMock)
    @patch('stresser.time.time')
    @patch('builtins.print')
    async def test_statistics_aggregation(
        self,
        mock_print,
        mock_time,
        mock_run_test,
        mock_get_models,
        tmp_path
    ):
        """Test that statistics are correctly aggregated."""
        # Create test conversation file
        conv_file = tmp_path / "test_conv.json"
        conv_file.write_text('{"conversation_id": "test", "messages": []}')

        # Mock time (3 iterations)
        start_time = 1000.0
        mock_time.side_effect = [
            start_time,
            start_time + 0.5,
            start_time + 1.0,
            start_time + 1.5,
            start_time + 3.0,  # Exit
        ]

        # Create different stats for each run
        stats1 = ConversationStats(conversation_id='test1')
        stats1.total_messages_sent = 5
        stats1.total_tokens = 100
        stats1.total_tokens_input = 40
        stats1.total_tokens_output = 60
        stats1.message_latencies = [1.0, 2.0]

        stats2 = ConversationStats(conversation_id='test2')
        stats2.total_messages_sent = 3
        stats2.total_tokens = 50
        stats2.total_tokens_input = 20
        stats2.total_tokens_output = 30
        stats2.message_latencies = [1.5]

        stats3 = ConversationStats(conversation_id='test3')
        stats3.total_messages_sent = 7
        stats3.total_tokens = 150
        stats3.total_tokens_input = 60
        stats3.total_tokens_output = 90
        stats3.message_latencies = [0.5, 1.0, 2.5]

        mock_run_test.side_effect = [stats1, stats2, stats3]

        # Run session
        result = await simulate_user_session(
            conversations_dir=str(tmp_path),
            duration_seconds=2,
            model_name='test-model',
            api_key='test_key',
            api_base='https://test.api'
        )

        # Verify aggregation
        assert result.total_conversations == 3
        assert result.total_messages_sent == 15  # 5 + 3 + 7
        assert result.total_tokens == 300  # 100 + 50 + 150
        assert result.total_tokens_input == 120  # 40 + 20 + 60
        assert result.total_tokens_output == 180  # 60 + 30 + 90

        # Verify latency calculations
        # All latencies: [1.0, 2.0, 1.5, 0.5, 1.0, 2.5]
        # Average: 8.5 / 6 = 1.417
        assert result.min_latency_seconds == 0.5
        assert result.max_latency_seconds == 2.5
        assert abs(result.average_latency_seconds - 1.417) < 0.01

    @pytest.mark.asyncio
    @patch('stresser.get_available_models')
    @patch('stresser.time.time')
    @patch('builtins.print')
    async def test_no_available_models(self, mock_print, mock_time, mock_get_models, tmp_path):
        """Test that ValueError is raised when no models available."""
        # Create test conversation file
        conv_file = tmp_path / "test_conv.json"
        conv_file.write_text('{"conversation_id": "test", "messages": []}')

        # Mock no models available
        mock_get_models.return_value = []

        with pytest.raises(ValueError, match="Could not determine available models"):
            await simulate_user_session(
                conversations_dir=str(tmp_path),
                duration_seconds=1,
                model_name=None,
                api_key='test_key',
                api_base='https://test.api'
            )


class TestPrintSessionReport:
    """Tests for print_session_report function."""

    @patch('builtins.print')
    def test_report_format(self, mock_print):
        """Test that report is formatted correctly."""
        # Create mock session stats
        stats = UserSessionStats(
            session_start=datetime(2025, 1, 1, 10, 0, 0),
            session_end=datetime(2025, 1, 1, 10, 1, 0),
            duration_seconds=60,
            total_conversations=10,
            successful_conversations=9,
            failed_conversations=1,
            total_messages_sent=50,
            total_tokens=1000,
            total_tokens_input=400,
            total_tokens_output=600,
            average_latency_seconds=1.5,
            min_latency_seconds=0.5,
            max_latency_seconds=3.0,
            conversations_per_minute=10.0,
            models_used=['model-a', 'model-b'],
            conversation_stats=[]
        )

        print_session_report(stats)

        # Verify print was called multiple times
        assert mock_print.call_count > 10

        # Verify key information was printed
        all_output = ' '.join([str(call.args[0]) if call.args else '' for call in mock_print.call_args_list])
        assert 'USER SESSION SUMMARY REPORT' in all_output
        assert '60s' in all_output
        assert 'Total Conversations: 10' in all_output
        assert 'Successful: 9' in all_output
        assert 'Failed: 1' in all_output
        assert '1,000' in all_output  # Formatted tokens

    @patch('builtins.print')
    def test_report_with_errors(self, mock_print):
        """Test that errors are included in report."""
        # Create failed conversation stats
        failed_stat1 = ConversationStats(conversation_id='test1')
        failed_stat1.error = "Error message 1"

        failed_stat2 = ConversationStats(conversation_id='test2')
        failed_stat2.error = "Error message 1"  # Same error

        failed_stat3 = ConversationStats(conversation_id='test3')
        failed_stat3.error = "Error message 2"

        stats = UserSessionStats(
            session_start=datetime(2025, 1, 1, 10, 0, 0),
            session_end=datetime(2025, 1, 1, 10, 1, 0),
            duration_seconds=60,
            total_conversations=3,
            successful_conversations=0,
            failed_conversations=3,
            total_messages_sent=0,
            total_tokens=0,
            total_tokens_input=0,
            total_tokens_output=0,
            average_latency_seconds=0,
            min_latency_seconds=0,
            max_latency_seconds=0,
            conversations_per_minute=3.0,
            models_used=['model-a'],
            conversation_stats=[failed_stat1, failed_stat2, failed_stat3]
        )

        print_session_report(stats)

        # Verify errors section was printed
        all_output = ' '.join([str(call.args[0]) if call.args else '' for call in mock_print.call_args_list])
        assert 'ERRORS' in all_output
        assert '[2x]' in all_output  # Error 1 appeared twice
        assert '[1x]' in all_output  # Error 2 appeared once
