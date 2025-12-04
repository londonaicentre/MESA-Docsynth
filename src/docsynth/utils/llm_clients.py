import logging
from abc import ABC, abstractmethod
from typing import Any, cast

from litellm import Choices, ModelResponse

from utils.llm import LLM
from utils.aws import AWS


class LLMClient(ABC):
    """
    LLM client abstractions for calling different API providers

    Args:
        model (str): Model name (e.g., 'gemini-2.5-flash')
        temperature (float, optional): Sampling temperature
        max_tokens (int, optional): Max tokens to generate
        api_key (str, optional): API key to authenticate to remove
            provider

    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 4000,
        api_key: str = "",
    ):
        self._logger: logging.Logger = logging.getLogger(__name__)
        self.api_key: str = api_key

        # Store generation parameters for API calls
        self.model_name: str = model
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self._logger.info(
            f"Initialized client with model={model}, temperature={temperature}, max_tokens={max_tokens}"
        )

    @abstractmethod
    def generate(self, prompt: str) -> str | None:
        """
        Generate a response from the LLM.

        Args:
            prompt (str): Prompt to send to the LLM

        Returns:
            str: Raw response text from the LLM or None
        """
        pass


class GeminiClient(LLMClient):
    """Client for Google Gemini API"""

    def init(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 4000,
        api_key: str = "",
    ) -> None:
        super().__init__(model, temperature, max_tokens, api_key)
        try:
            import google.generativeai as genai

            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

    def generate(self, prompt: str) -> str | None:
        """Generate response from Gemini."""
        self._logger.debug(f"Sending prompt to Gemini (length={len(prompt)} chars)")

        try:
            # Disable all safety filters to allow medical/technical content generation
            safety_settings: list[dict[str, Any]] = [
                {
                    "category": self.genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": self.genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": self.genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": self.genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": self.genai.types.HarmBlockThreshold.BLOCK_NONE,
                },
            ]

            response: ModelResponse | None = LLM.completion(
                "gemini/" + self.model_name,
                None,
                prompt,
                self.api_key,
                self.max_tokens,
                self.temperature,
                safety_settings=safety_settings,
            )
            if response is not None:
                # Check if response was blocked by safety filters
                if not cast(Choices, response.choices[0]).message.content:
                    finish_reason: str | None = response.choices[0].finish_reason
                    self._logger.error(
                        f"Gemini blocked response. Finish reason: {finish_reason}"
                    )
                    raise ValueError(
                        f"Response blocked by Gemini. Finish reason: {finish_reason}"
                    )

                result: str | None = cast(Choices, response.choices[0]).message.content
                if result is not None:
                    self._logger.debug(
                        f"Received response from Gemini (length={len(result)} chars)"
                    )
                    return result
            return None
        except Exception as e:
            self._logger.error(f"Error generating from Gemini: {e}")
            raise


class AnthropicClient(LLMClient):
    """Client for Anthropic API"""

    def generate(self, prompt: str) -> str | None:
        """Generate response from Claude"""
        self._logger.debug(f"Sending prompt to Claude (length={len(prompt)} chars)")

        try:
            response: ModelResponse | None = AWS.bedrock_completion(
                self.model_name,
                None,
                prompt,
                self.api_key,
                self.max_tokens,
                self.temperature,
            )
            if response is not None:
                if not cast(Choices, response.choices[0]).message.content:
                    raise ValueError("Response not provided by Bedrock.")

                result: str | None = cast(Choices, response.choices[0]).message.content
                if result is not None:
                    self._logger.debug(
                        f"Received response from Claude (length={len(result)} chars)"
                    )
                    return result
            return None
        except Exception as e:
            self._logger.error(f"Error generating from Claude: {e}")
            raise


class LocalClient(LLMClient):
    """Client for local OpenAI-compatible endpoint"""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 4000,
    ):
        """
        Initialize local OpenAI-compatible client.

        Args:
            base_url (str): Base URL for the API (e.g., 'http://localhost:1234/v1')

        """
        super().__init__(model, temperature, max_tokens)
        self.base_url: str = base_url

    def generate(self, prompt: str) -> str | None:
        """
        Generate response from local API.
        """

        self._logger.debug(f"Sending prompt to local API (length={len(prompt)} chars)")

        try:
            response: ModelResponse | None = LLM.completion(
                self.model_name,
                None,
                prompt,
                self.api_key,
                self.max_tokens,
                self.temperature,
                api_base=self.base_url,
            )
            if response is not None:
                if not cast(Choices, response.choices[0]).message.content:
                    raise ValueError("Response not provided by local client.")

                result: str | None = cast(Choices, response.choices[0]).message.content
                if result is not None:
                    self._logger.debug(
                        f"Received response from local API (length={len(result)} chars)"
                    )
                    return result
            return None
        except Exception as e:
            self._logger.error(f"Error generating from local API: {e}")
            raise
