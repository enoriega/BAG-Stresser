import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage

from stresser import (
    run_conversation_stress_test,
    calculate_sleep_time,
    ConversationStats
)


@pytest.fixture
def sample_conversation_file(tmp_path):
    """Create a sample conversation JSON file for testing."""
    conversation_data = {
        "conversation_id": "test_conv_01",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you for asking!"
            },
            {
                "role": "user",
                "content": "Can you help me with Python?"
            },
            {
                "role": "assistant",
                "content": "Of course! I'd be happy to help you with Python."
            },
            {
                "role": "user",
                "content": "Great! How do I create a list?"
            },
            {
                "role": "assistant",
                "content": "You can create a list in Python using square brackets: my_list = [1, 2, 3]"
            }
        ]
    }

    file_path = tmp_path / "test_conversation.json"
    with open(file_path, 'w') as f:
        json.dump(conversation_data, f)

    return file_path


@pytest.fixture
def sample_single_message_conversation(tmp_path):
    """Create a conversation with just one message."""
    conversation_data = {
        "conversation_id": "test_conv_single",
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2?"
            },
            {
                "role": "assistant",
                "content": "2+2 equals 4."
            }
        ]
    }

    file_path = tmp_path / "single_message.json"
    with open(file_path, 'w') as f:
        json.dump(conversation_data, f)

    return file_path


class TestCalculateSleepTime:
    """Tests for the calculate_sleep_time function."""

    def test_base_time_with_zero_length(self):
        """Test that base time is returned for zero-length message."""
        sleep_time = calculate_sleep_time(0, base_time=1.0, time_per_char=0.01)
        # Should be around 1.0 with ±20% randomness
        assert 0.8 <= sleep_time <= 1.2

    def test_sleep_time_increases_with_message_length(self):
        """Test that sleep time increases with message length."""
        # Use same random seed for consistency
        short_times = [calculate_sleep_time(10) for _ in range(100)]
        long_times = [calculate_sleep_time(100) for _ in range(100)]

        avg_short = sum(short_times) / len(short_times)
        avg_long = sum(long_times) / len(long_times)

        assert avg_long > avg_short

    def test_sleep_time_has_randomness(self):
        """Test that sleep time includes randomness."""
        times = [calculate_sleep_time(50) for _ in range(10)]
        # Not all times should be identical
        assert len(set(times)) > 1


class TestRunConversationStressTest:
    """Tests for the run_conversation_stress_test function."""

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            run_conversation_stress_test(
                conversation_file_path="/nonexistent/file.json",
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=100
            )

    @patch('stresser.ChatOpenAI')
    @patch('stresser.time.sleep')
    def test_successful_conversation(self, mock_sleep, mock_chat_openai, sample_conversation_file):
        """Test a successful conversation with all messages."""
        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        # Mock responses
        mock_responses = []
        for i in range(3):
            mock_response = Mock(spec=AIMessage)
            mock_response.content = f"Mock response {i+1}"
            mock_response.response_metadata = {
                'token_usage': {
                    'prompt_tokens': 10 + i,
                    'completion_tokens': 5 + i,
                    'total_tokens': 15 + (i*2)
                }
            }
            mock_responses.append(mock_response)

        mock_llm_instance.invoke.side_effect = mock_responses

        # Run the test
        stats = run_conversation_stress_test(
            conversation_file_path=str(sample_conversation_file),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_key="test_key"
        )

        # Assertions
        assert stats.conversation_id == "test_conv_01"
        assert stats.total_user_messages == 3
        assert stats.total_ai_responses == 3
        assert stats.total_messages_sent == 3
        assert stats.error is None
        assert len(stats.message_latencies) == 3
        assert stats.total_tokens_input == 33  # 10 + 11 + 12
        assert stats.total_tokens_output == 18  # 5 + 6 + 7
        assert stats.average_latency_seconds >= 0
        assert mock_llm_instance.invoke.call_count == 3
        # Should sleep 2 times (between 3 messages)
        assert mock_sleep.call_count == 2

    @patch('stresser.ChatOpenAI')
    @patch('stresser.time.sleep')
    def test_max_messages_limit(self, mock_sleep, mock_chat_openai, sample_conversation_file):
        """Test that max_messages parameter limits the number of messages sent."""
        mock_llm_instance = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        # Mock response
        mock_response = Mock(spec=AIMessage)
        mock_response.content = "Mock response"
        mock_response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 10,
                'completion_tokens': 5,
                'total_tokens': 15
            }
        }
        mock_llm_instance.invoke.return_value = mock_response

        # Run with max_messages=2
        stats = run_conversation_stress_test(
            conversation_file_path=str(sample_conversation_file),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            max_messages=2,
            api_key="test_key"
        )

        # Should only send 2 messages
        assert stats.total_user_messages == 2
        assert stats.total_ai_responses == 2
        assert mock_llm_instance.invoke.call_count == 2
        # Should sleep 1 time (between 2 messages)
        assert mock_sleep.call_count == 1

    @patch('stresser.ChatOpenAI')
    @patch('stresser.time.sleep')
    def test_single_message_conversation(self, mock_sleep, mock_chat_openai, sample_single_message_conversation):
        """Test conversation with only one message."""
        mock_llm_instance = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        mock_response = Mock(spec=AIMessage)
        mock_response.content = "4"
        mock_response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 5,
                'completion_tokens': 1,
                'total_tokens': 6
            }
        }
        mock_llm_instance.invoke.return_value = mock_response

        stats = run_conversation_stress_test(
            conversation_file_path=str(sample_single_message_conversation),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_key="test_key"
        )

        assert stats.total_user_messages == 1
        assert stats.total_ai_responses == 1
        assert mock_llm_instance.invoke.call_count == 1
        # Should not sleep after the last message
        assert mock_sleep.call_count == 0

    @patch('stresser.ChatOpenAI')
    def test_llm_initialization_error(self, mock_chat_openai, sample_conversation_file):
        """Test handling of LLM initialization error."""
        mock_chat_openai.side_effect = Exception("API key invalid")

        stats = run_conversation_stress_test(
            conversation_file_path=str(sample_conversation_file),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_key="invalid_key"
        )

        assert stats.error is not None
        assert "Error initializing LLM" in stats.error
        assert stats.total_messages_sent == 0

    @patch('stresser.ChatOpenAI')
    @patch('stresser.time.sleep')
    def test_llm_invocation_error(self, mock_sleep, mock_chat_openai, sample_conversation_file):
        """Test handling of error during LLM invocation."""
        mock_llm_instance = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        # First call succeeds, second call fails
        mock_response = Mock(spec=AIMessage)
        mock_response.content = "First response"
        mock_response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 10,
                'completion_tokens': 5,
                'total_tokens': 15
            }
        }
        mock_llm_instance.invoke.side_effect = [
            mock_response,
            Exception("Rate limit exceeded")
        ]

        stats = run_conversation_stress_test(
            conversation_file_path=str(sample_conversation_file),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_key="test_key"
        )

        # Should have processed first message successfully
        assert stats.total_user_messages == 1
        assert stats.total_ai_responses == 1
        assert stats.error is not None
        assert "Error during message 2" in stats.error

    @patch('stresser.ChatOpenAI')
    @patch('stresser.time.sleep')
    def test_statistics_calculation(self, mock_sleep, mock_chat_openai, sample_conversation_file):
        """Test that statistics are calculated correctly."""
        mock_llm_instance = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        # Create mock responses with different latencies
        mock_responses = []
        for i in range(3):
            mock_response = Mock(spec=AIMessage)
            mock_response.content = f"Response {i+1}"
            mock_response.response_metadata = {
                'token_usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15
                }
            }
            mock_responses.append(mock_response)

        mock_llm_instance.invoke.side_effect = mock_responses

        stats = run_conversation_stress_test(
            conversation_file_path=str(sample_conversation_file),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_key="test_key"
        )

        # Check statistics
        assert stats.total_latency_seconds > 0
        assert stats.average_latency_seconds == stats.total_latency_seconds / 3
        assert stats.min_latency_seconds <= stats.max_latency_seconds
        assert stats.min_latency_seconds > 0
        assert len(stats.message_latencies) == 3

    @patch('stresser.ChatOpenAI')
    @patch('stresser.time.sleep')
    def test_conversation_history_building(self, mock_sleep, mock_chat_openai, sample_conversation_file):
        """Test that conversation history is built correctly."""
        mock_llm_instance = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        # Track the history lengths at each call
        history_lengths = []

        def capture_history_length(messages):
            history_lengths.append(len(messages))
            mock_response = Mock(spec=AIMessage)
            mock_response.content = "Response"
            mock_response.response_metadata = {
                'token_usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15
                }
            }
            return mock_response

        mock_llm_instance.invoke.side_effect = capture_history_length

        run_conversation_stress_test(
            conversation_file_path=str(sample_conversation_file),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            api_key="test_key"
        )

        # Check that invoke was called 3 times (3 user messages in sample)
        assert mock_llm_instance.invoke.call_count == 3

        # Verify that conversation history grows with each call
        # First call: 1 message (user1)
        # Second call: 3 messages (user1, ai1_from_json, user2)
        # Third call: 5 messages (user1, ai1_from_json, user2, ai2_from_json, user3)
        assert len(history_lengths) == 3
        assert history_lengths[0] == 1
        assert history_lengths[1] == 3
        assert history_lengths[2] == 5

    @patch('stresser.ChatOpenAI')
    @patch('stresser.time.sleep')
    def test_max_messages_zero(self, mock_sleep, mock_chat_openai, sample_conversation_file):
        """Test that max_messages=0 sends no messages."""
        mock_llm_instance = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        stats = run_conversation_stress_test(
            conversation_file_path=str(sample_conversation_file),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            max_messages=0,
            api_key="test_key"
        )

        assert stats.total_messages_sent == 0
        assert mock_llm_instance.invoke.call_count == 0

    @patch('stresser.ChatOpenAI')
    @patch('stresser.time.sleep')
    def test_api_parameters_passed(self, mock_sleep, mock_chat_openai, sample_conversation_file):
        """Test that API parameters are correctly passed to ChatOpenAI."""
        mock_llm_instance = Mock()
        mock_chat_openai.return_value = mock_llm_instance

        mock_response = Mock(spec=AIMessage)
        mock_response.content = "Response"
        mock_response.response_metadata = {'token_usage': {}}
        mock_llm_instance.invoke.return_value = mock_response

        run_conversation_stress_test(
            conversation_file_path=str(sample_conversation_file),
            model_name="gpt-4",
            temperature=0.9,
            max_tokens=200,
            max_messages=1,
            api_key="custom_key",
            api_base="https://custom.api.com"
        )

        # Verify ChatOpenAI was called with correct parameters
        mock_chat_openai.assert_called_once_with(
            model="gpt-4",
            temperature=0.9,
            max_tokens=200,
            api_key="custom_key",
            base_url="https://custom.api.com"
        )
