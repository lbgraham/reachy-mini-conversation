"""Google Cloud Text-to-Speech wrapper for Reachy Mini.

Provides a simple interface for converting text to speech audio
compatible with Reachy Mini's audio system (24kHz PCM).
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from google.cloud import texttospeech

logger = logging.getLogger(__name__)

# Reachy Mini uses 24kHz audio
OUTPUT_SAMPLE_RATE = 24000


class GoogleTTS:
    """Google Cloud Text-to-Speech wrapper.

    Converts text to PCM audio at 24kHz for Reachy Mini compatibility.
    """

    def __init__(
        self,
        voice_name: str = "en-US-Neural2-F",
        language_code: str = "en-US",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ):
        """Initialize Google TTS client.

        Args:
            voice_name: Google TTS voice name (e.g., "en-US-Neural2-F")
            language_code: Language code (e.g., "en-US")
            speaking_rate: Speech speed multiplier (0.25 to 4.0)
            pitch: Pitch adjustment in semitones (-20.0 to 20.0)
        """
        self.client = texttospeech.TextToSpeechClient()
        self.voice_name = voice_name
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        self.pitch = pitch

        # Configure voice
        self.voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )

        # Configure audio output - LINEAR16 is PCM
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=OUTPUT_SAMPLE_RATE,
            speaking_rate=speaking_rate,
            pitch=pitch,
        )

        logger.info(
            "GoogleTTS initialized: voice=%s, rate=%s, pitch=%s",
            voice_name, speaking_rate, pitch
        )

    def synthesize(self, text: str) -> NDArray[np.int16]:
        """Convert text to speech audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio as int16 numpy array at 24kHz
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return np.array([], dtype=np.int16)

        # Build synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)

        try:
            # Perform synthesis
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config,
            )

            # Convert bytes to int16 numpy array
            # LINEAR16 encoding returns raw PCM bytes (little-endian int16)
            audio_data = np.frombuffer(response.audio_content, dtype=np.int16)

            logger.debug(
                "Synthesized %d chars -> %d samples (%.2fs)",
                len(text),
                len(audio_data),
                len(audio_data) / OUTPUT_SAMPLE_RATE
            )

            return audio_data

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            raise

    def synthesize_ssml(self, ssml: str) -> NDArray[np.int16]:
        """Convert SSML to speech audio.

        Args:
            ssml: SSML markup to synthesize

        Returns:
            Audio as int16 numpy array at 24kHz
        """
        if not ssml or not ssml.strip():
            logger.warning("Empty SSML provided to TTS")
            return np.array([], dtype=np.int16)

        synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

        try:
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config,
            )

            audio_data = np.frombuffer(response.audio_content, dtype=np.int16)

            logger.debug(
                "Synthesized SSML -> %d samples (%.2fs)",
                len(audio_data),
                len(audio_data) / OUTPUT_SAMPLE_RATE
            )

            return audio_data

        except Exception as e:
            logger.error("TTS SSML synthesis failed: %s", e)
            raise

    def get_sample_rate(self) -> int:
        """Return the output sample rate."""
        return OUTPUT_SAMPLE_RATE

    def set_voice(self, voice_name: str, language_code: Optional[str] = None) -> None:
        """Change the voice.

        Args:
            voice_name: New voice name
            language_code: New language code (optional, keeps current if None)
        """
        if language_code:
            self.language_code = language_code
        self.voice_name = voice_name

        self.voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=voice_name,
        )

        logger.info("Voice changed to: %s (%s)", voice_name, self.language_code)

    def set_speaking_rate(self, rate: float) -> None:
        """Change the speaking rate.

        Args:
            rate: New speaking rate (0.25 to 4.0)
        """
        self.speaking_rate = max(0.25, min(4.0, rate))
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=OUTPUT_SAMPLE_RATE,
            speaking_rate=self.speaking_rate,
            pitch=self.pitch,
        )
        logger.info("Speaking rate changed to: %s", self.speaking_rate)

    @staticmethod
    def list_voices(language_code: Optional[str] = None) -> list[dict]:
        """List available voices.

        Args:
            language_code: Filter by language (optional)

        Returns:
            List of voice info dicts
        """
        client = texttospeech.TextToSpeechClient()
        response = client.list_voices(language_code=language_code)

        voices = []
        for voice in response.voices:
            voices.append({
                "name": voice.name,
                "language_codes": list(voice.language_codes),
                "gender": texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                "natural_sample_rate": voice.natural_sample_rate_hertz,
            })

        return voices
