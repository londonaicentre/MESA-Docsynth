import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

"""
llm_clients.py - LLM client abstractions for calling different API providers.
"""

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract base class for any clients
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt:
                Prompt to send to the LLM

        Returns:
            Raw response text from the LLM
        """
        pass


class GeminiClient(LLMClient):
    """
    Client for Google Gemini API
    """

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 8000):
        """
        Initialise Gemini client.

        Args:
            model:
                Model name (e.g., 'gemini-2.5-flash')
            temperature:
                Sampling temperature
            max_tokens:
                Max tokens to generate
        """
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.genai.configure(api_key=api_key)

        # Store generation parameters for API calls
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = self.genai.GenerativeModel(model)

        logger.info(
            f"Initialized GeminiClient with model={model}, temperature={temperature}, max_tokens={max_tokens}"
        )

    def generate(self, prompt: str) -> str:
        """
        Generate response from Gemini.
        """
        logger.debug(f"Sending prompt to Gemini (length={len(prompt)} chars)")

        try:
            # Disable all safety filters to allow medical/technical content generation
            safety_settings = [
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

            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
                safety_settings=safety_settings,
            )

            # Check if response was blocked by safety filters
            if not response.parts:
                finish_reason = (
                    response.candidates[0].finish_reason
                    if response.candidates
                    else None
                )
                logger.error(f"Gemini blocked response. Finish reason: {finish_reason}")
                raise ValueError(
                    f"Response blocked by Gemini. Finish reason: {finish_reason}"
                )

            result = response.text
            logger.debug(f"Received response from Gemini (length={len(result)} chars)")
            return result

        except Exception as e:
            logger.error(f"Error generating from Gemini: {e}")
            raise


class ClaudeClient(LLMClient):
    """
    Client for Anthropic Claude API
    """

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 4000):
        """
        Initialize Claude client.

        Args:
            model:
                Model name (e.g., 'claude-sonnet-4-5-20250929')
            temperature:
                Sampling temperature
            max_tokens:
                Max tokens to generate
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            f"Initialized ClaudeClient with model={model}, temperature={temperature}, max_tokens={max_tokens}"
        )

    def generate(self, prompt: str) -> str:
        """
        Generate response from Claude.
        """
        logger.debug(f"Sending prompt to Claude (length={len(prompt)} chars)")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.content[0].text
            logger.debug(f"Received response from Claude (length={len(result)} chars)")
            return result

        except Exception as e:
            logger.error(f"Error generating from Claude: {e}")
            raise


class LocalClient(LLMClient):
    """
    Client for local OpenAI-compatible endpoint
    """

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
            base_url:
                Base URL for the API (e.g., 'http://localhost:1234/v1')
            model:
                Model name
            temperature:
                Sampling temperature
            max_tokens:
                Max tokens to generate
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client with local endpoint (API key not required)
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
        )

        logger.info(
            f"Initialised LocalClient with base_url={base_url}, model={model}, temperature={temperature}, max_tokens={max_tokens}"
        )

    def generate(self, prompt: str) -> str:
        """
        Generate response from local API.
        """
        logger.debug(f"Sending prompt to local API (length={len(prompt)} chars)")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            result = response.choices[0].message.content
            logger.debug(
                f"Received response from local API (length={len(result)} chars)"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating from local API: {e}")
            raise


def create_llm_client(llm_config: dict) -> Optional[LLMClient]:
    """
    Factory function to create the appropriate LLM client based on config.

    Args:
        llm_config:
            Dictionary containing LLM configuration from pipeline.yml

    Returns:
        LLMClient instance or None if disabled
    """
    if not llm_config.get("enabled", False):
        logger.info("LLM generation disabled")
        return None

    provider = llm_config.get("provider", "none")

    # Return None if no provider configured
    if provider == "none":
        logger.info("LLM provider set to 'none'")
        return None

    elif provider == "gemini":
        config = llm_config["gemini"]
        return GeminiClient(
            model=config["model"],
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_tokens", 4000),
        )

    elif provider == "claude":
        config = llm_config["claude"]
        return ClaudeClient(
            model=config["model"],
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_tokens", 4000),
        )

    elif provider == "local":
        config = llm_config["local"]
        # Read base_url and model from environment variables
        base_url = os.getenv("LOCAL_LLM_BASE_URL")
        model = os.getenv("LOCAL_LLM_MODEL")

        if not base_url:
            raise ValueError("LOCAL_LLM_BASE_URL not found in environment variables")
        if not model:
            raise ValueError("LOCAL_LLM_MODEL not found in environment variables")

        return LocalClient(
            base_url=base_url,
            model=model,
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_tokens", 4000),
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
