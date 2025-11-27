import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from multi_session import (
    run_multi_session,
    run_single_session,
    aggregate_session_stats,
    print_aggregate_report
)
from stresser import UserSessionStats, ConversationStats


class TestRunSingleSession:
    """Tests for run_single_session function."""

    @pytest.mark.asyncio
    @patch('multi_session.simulate_user_session', new_callable=AsyncMock)
    @patch('builtins.print')
    async def test_successful_single_session(self, mock_print, mock_simulate):
        """Test a successful single session execution."""
        # Create mock stats
        mock_stats = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=30,
            total_conversations=5,
            successful_conversations=5,
            failed_conversations=0,
            total_messages_sent=25,
            total_tokens=1000,
            total_tokens_input=400,
            total_tokens_output=600,
            average_latency_seconds=1.5,
            min_latency_seconds=0.5,
            max_latency_seconds=3.0,
            conversations_per_minute=10.0,
            models_used=['gpt-3.5-turbo'],
            conversation_stats=[]
        )
        mock_simulate.return_value = mock_stats

        # Run single session
        result = await run_single_session(
            session_id=1,
            duration=30,
            conversations_dir='conversations',
            model_name='gpt-3.5-turbo',
            api_key='test_key',
            api_base='https://test.api',
            temperature_range=(0.5, 1.0)
        )

        # Verify result
        assert result == mock_stats
        mock_simulate.assert_called_once()

        # Verify logging
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('[Session 1]' in str(call) for call in print_calls)

    @pytest.mark.asyncio
    @patch('multi_session.simulate_user_session', new_callable=AsyncMock)
    @patch('builtins.print')
    async def test_failed_single_session(self, mock_print, mock_simulate):
        """Test a single session that fails with an exception."""
        # Mock simulate_user_session to raise exception
        mock_simulate.side_effect = Exception("API Error")

        # Run single session and expect exception to be raised
        with pytest.raises(Exception, match="API Error"):
            await run_single_session(
                session_id=2,
                duration=30,
                conversations_dir='conversations',
                model_name='gpt-4',
                api_key='test_key',
                api_base='https://test.api',
                temperature_range=(0.5, 1.0)
            )

        # Verify error logging
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('[Session 2]' in str(call) and 'Failed' in str(call) for call in print_calls)


class TestRunMultiSession:
    """Tests for run_multi_session function."""

    @pytest.mark.asyncio
    @patch('multi_session.run_single_session', new_callable=AsyncMock)
    @patch('builtins.print')
    async def test_single_session_execution(self, mock_print, mock_run_single):
        """Test running a single session (default)."""
        # Create mock stats
        mock_stats = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=30,
            total_conversations=5,
            successful_conversations=5,
            failed_conversations=0,
            total_messages_sent=25,
            total_tokens=1000,
            total_tokens_input=400,
            total_tokens_output=600,
            average_latency_seconds=1.5,
            min_latency_seconds=0.5,
            max_latency_seconds=3.0,
            conversations_per_minute=10.0,
            models_used=['gpt-3.5-turbo'],
            conversation_stats=[]
        )
        mock_run_single.return_value = mock_stats

        # Run multi-session with 1 session
        results = await run_multi_session(
            num_sessions=1,
            duration=30,
            conversations_dir='conversations',
            model_name='gpt-3.5-turbo',
            api_key='test_key',
            api_base='https://test.api',
            temperature_range=(0.5, 1.0)
        )

        # Verify results
        assert len(results) == 1
        assert results[0] == mock_stats
        mock_run_single.assert_called_once()

    @pytest.mark.asyncio
    @patch('multi_session.run_single_session', new_callable=AsyncMock)
    @patch('builtins.print')
    async def test_multiple_concurrent_sessions(self, mock_print, mock_run_single):
        """Test running multiple concurrent sessions."""
        # Create different mock stats for each session
        mock_stats_list = []
        for i in range(3):
            stats = UserSessionStats(
                session_start=datetime.now(),
                session_end=datetime.now(),
                duration_seconds=30,
                total_conversations=5 + i,
                successful_conversations=5 + i,
                failed_conversations=0,
                total_messages_sent=25 + i * 5,
                total_tokens=1000 + i * 100,
                total_tokens_input=400 + i * 40,
                total_tokens_output=600 + i * 60,
                average_latency_seconds=1.5,
                min_latency_seconds=0.5,
                max_latency_seconds=3.0,
                conversations_per_minute=10.0,
                models_used=['gpt-3.5-turbo'],
                conversation_stats=[]
            )
            mock_stats_list.append(stats)

        mock_run_single.side_effect = mock_stats_list

        # Run multi-session with 3 sessions
        results = await run_multi_session(
            num_sessions=3,
            duration=30,
            conversations_dir='conversations',
            model_name='gpt-3.5-turbo',
            api_key='test_key',
            api_base='https://test.api',
            temperature_range=(0.5, 1.0)
        )

        # Verify results
        assert len(results) == 3
        assert results == mock_stats_list
        assert mock_run_single.call_count == 3

        # Verify concurrent execution (all sessions started)
        print_output = ' '.join([str(call) for call in mock_print.call_args_list])
        assert 'Launching all 3 sessions concurrently' in print_output

    @pytest.mark.asyncio
    @patch('multi_session.run_single_session', new_callable=AsyncMock)
    @patch('builtins.print')
    async def test_some_sessions_fail(self, mock_print, mock_run_single):
        """Test handling when some sessions fail."""
        # Mock: first succeeds, second fails, third succeeds
        mock_stats1 = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=30,
            total_conversations=5,
            successful_conversations=5,
            failed_conversations=0,
            total_messages_sent=25,
            total_tokens=1000,
            total_tokens_input=400,
            total_tokens_output=600,
            average_latency_seconds=1.5,
            min_latency_seconds=0.5,
            max_latency_seconds=3.0,
            conversations_per_minute=10.0,
            models_used=['gpt-3.5-turbo'],
            conversation_stats=[]
        )

        mock_stats3 = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=30,
            total_conversations=6,
            successful_conversations=6,
            failed_conversations=0,
            total_messages_sent=30,
            total_tokens=1200,
            total_tokens_input=480,
            total_tokens_output=720,
            average_latency_seconds=1.5,
            min_latency_seconds=0.5,
            max_latency_seconds=3.0,
            conversations_per_minute=12.0,
            models_used=['gpt-4'],
            conversation_stats=[]
        )

        mock_run_single.side_effect = [
            mock_stats1,
            Exception("Session 2 failed"),
            mock_stats3
        ]

        # Run multi-session with 3 sessions
        results = await run_multi_session(
            num_sessions=3,
            duration=30,
            conversations_dir='conversations',
            model_name=None,
            api_key='test_key',
            api_base='https://test.api',
            temperature_range=(0.5, 1.0)
        )

        # Verify only successful results returned
        assert len(results) == 2
        assert results[0] == mock_stats1
        assert results[1] == mock_stats3

        # Verify failure was logged
        print_output = ' '.join([str(call) for call in mock_print.call_args_list])
        assert 'FAILED' in print_output or 'Failed sessions: 1' in print_output


class TestAggregateSessionStats:
    """Tests for aggregate_session_stats function."""

    def test_empty_sessions_list(self):
        """Test aggregation with empty list."""
        result = aggregate_session_stats([])
        assert result == {}

    def test_single_session_aggregation(self):
        """Test aggregation with a single session."""
        conv_stats = ConversationStats(conversation_id='conv1')
        conv_stats.message_latencies = [1.0, 2.0, 1.5]

        session = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=60,
            total_conversations=10,
            successful_conversations=9,
            failed_conversations=1,
            total_messages_sent=50,
            total_tokens=2000,
            total_tokens_input=800,
            total_tokens_output=1200,
            average_latency_seconds=1.5,
            min_latency_seconds=1.0,
            max_latency_seconds=2.0,
            conversations_per_minute=10.0,
            models_used=['gpt-3.5-turbo'],
            conversation_stats=[conv_stats]
        )

        result = aggregate_session_stats([session])

        assert result['num_sessions'] == 1
        assert result['total_conversations'] == 10
        assert result['successful_conversations'] == 9
        assert result['failed_conversations'] == 1
        assert result['total_messages'] == 50
        assert result['total_tokens'] == 2000
        assert result['total_tokens_input'] == 800
        assert result['total_tokens_output'] == 1200
        assert result['avg_latency'] == 1.5
        assert result['min_latency'] == 1.0
        assert result['max_latency'] == 2.0
        assert result['models_used'] == ['gpt-3.5-turbo']
        assert result['total_duration'] == 60

    def test_multiple_sessions_aggregation(self):
        """Test aggregation with multiple sessions."""
        conv_stats1 = ConversationStats(conversation_id='conv1')
        conv_stats1.message_latencies = [1.0, 2.0]

        conv_stats2 = ConversationStats(conversation_id='conv2')
        conv_stats2.message_latencies = [1.5, 2.5]

        session1 = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=60,
            total_conversations=10,
            successful_conversations=9,
            failed_conversations=1,
            total_messages_sent=50,
            total_tokens=2000,
            total_tokens_input=800,
            total_tokens_output=1200,
            average_latency_seconds=1.5,
            min_latency_seconds=1.0,
            max_latency_seconds=2.0,
            conversations_per_minute=10.0,
            models_used=['gpt-3.5-turbo'],
            conversation_stats=[conv_stats1]
        )

        session2 = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=50,
            total_conversations=8,
            successful_conversations=7,
            failed_conversations=1,
            total_messages_sent=40,
            total_tokens=1600,
            total_tokens_input=640,
            total_tokens_output=960,
            average_latency_seconds=2.0,
            min_latency_seconds=1.5,
            max_latency_seconds=2.5,
            conversations_per_minute=9.6,
            models_used=['gpt-4'],
            conversation_stats=[conv_stats2]
        )

        result = aggregate_session_stats([session1, session2])

        assert result['num_sessions'] == 2
        assert result['total_conversations'] == 18
        assert result['successful_conversations'] == 16
        assert result['failed_conversations'] == 2
        assert result['total_messages'] == 90
        assert result['total_tokens'] == 3600
        assert result['total_tokens_input'] == 1440
        assert result['total_tokens_output'] == 2160
        # Average of all latencies: (1.0 + 2.0 + 1.5 + 2.5) / 4 = 1.75
        assert result['avg_latency'] == 1.75
        assert result['min_latency'] == 1.0
        assert result['max_latency'] == 2.5
        assert result['models_used'] == ['gpt-3.5-turbo', 'gpt-4']
        assert result['total_duration'] == 60  # max of 60 and 50

    def test_aggregation_with_mixed_models(self):
        """Test that models are correctly aggregated and sorted."""
        session1 = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=60,
            total_conversations=5,
            successful_conversations=5,
            failed_conversations=0,
            total_messages_sent=25,
            total_tokens=1000,
            total_tokens_input=400,
            total_tokens_output=600,
            average_latency_seconds=1.5,
            min_latency_seconds=1.0,
            max_latency_seconds=2.0,
            conversations_per_minute=5.0,
            models_used=['gpt-4', 'gpt-3.5-turbo'],
            conversation_stats=[]
        )

        session2 = UserSessionStats(
            session_start=datetime.now(),
            session_end=datetime.now(),
            duration_seconds=60,
            total_conversations=5,
            successful_conversations=5,
            failed_conversations=0,
            total_messages_sent=25,
            total_tokens=1000,
            total_tokens_input=400,
            total_tokens_output=600,
            average_latency_seconds=1.5,
            min_latency_seconds=1.0,
            max_latency_seconds=2.0,
            conversations_per_minute=5.0,
            models_used=['claude-3', 'gpt-4'],
            conversation_stats=[]
        )

        result = aggregate_session_stats([session1, session2])

        # Should have unique, sorted models
        assert result['models_used'] == ['claude-3', 'gpt-3.5-turbo', 'gpt-4']


class TestPrintAggregateReport:
    """Tests for print_aggregate_report function."""

    @patch('builtins.print')
    def test_report_format(self, mock_print):
        """Test that aggregate report is formatted correctly."""
        stats = {
            'num_sessions': 3,
            'total_conversations': 30,
            'successful_conversations': 28,
            'failed_conversations': 2,
            'total_messages': 150,
            'total_tokens': 6000,
            'total_tokens_input': 2400,
            'total_tokens_output': 3600,
            'avg_latency': 1.5,
            'min_latency': 0.5,
            'max_latency': 3.0,
            'models_used': ['gpt-3.5-turbo', 'gpt-4'],
            'total_duration': 60.0,
            'conversations_per_minute': 30.0,
            'tokens_per_second': 100.0
        }

        print_aggregate_report(stats)

        # Verify print was called multiple times
        assert mock_print.call_count > 15

        # Verify key information was printed
        all_output = ' '.join([str(call.args[0]) if call.args else '' for call in mock_print.call_args_list])
        assert 'AGGREGATE STATISTICS' in all_output
        assert 'Number of Sessions: 3' in all_output
        assert 'Total Conversations: 30' in all_output
        assert 'Successful: 28' in all_output
        assert 'Failed: 2' in all_output
        assert '6,000' in all_output  # Formatted tokens
        assert '100.0 tokens/second' in all_output

    @patch('builtins.print')
    def test_report_with_models(self, mock_print):
        """Test that models are printed in the report."""
        stats = {
            'num_sessions': 2,
            'total_conversations': 20,
            'successful_conversations': 20,
            'failed_conversations': 0,
            'total_messages': 100,
            'total_tokens': 4000,
            'total_tokens_input': 1600,
            'total_tokens_output': 2400,
            'avg_latency': 1.5,
            'min_latency': 0.5,
            'max_latency': 3.0,
            'models_used': ['claude-3', 'gpt-3.5-turbo', 'gpt-4'],
            'total_duration': 60.0,
            'conversations_per_minute': 20.0,
            'tokens_per_second': 66.67
        }

        print_aggregate_report(stats)

        # Verify models section
        all_output = ' '.join([str(call.args[0]) if call.args else '' for call in mock_print.call_args_list])
        assert 'MODELS USED' in all_output
        assert 'claude-3' in all_output
        assert 'gpt-3.5-turbo' in all_output
        assert 'gpt-4' in all_output
