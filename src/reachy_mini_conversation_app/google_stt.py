"""Google Cloud Speech-to-Text streaming wrapper for Reachy Mini.

Provides real-time speech recognition with interim results and
automatic end-of-speech detection.
"""

import asyncio
import logging
import queue
import threading
from typing import AsyncIterator, Callable, Optional
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample
from google.cloud import speech

logger = logging.getLogger(__name__)

# Google Cloud STT preferred sample rate
GOOGLE_SAMPLE_RATE = 16000


@dataclass
class TranscriptResult:
    """Represents a transcription result."""
    text: str
    is_final: bool
    confidence: float = 0.0
    stability: float = 0.0


class GoogleSTTStream:
    """Google Cloud Speech-to-Text streaming wrapper.

    Handles real-time speech recognition with:
    - Streaming audio input
    - Interim (partial) results for UI feedback
    - Final results with end-of-speech detection
    - Automatic session management
    """

    def __init__(
        self,
        language_code: str = "en-US",
        sample_rate: int = GOOGLE_SAMPLE_RATE,
        single_utterance: bool = True,
        interim_results: bool = True,
        on_transcript: Optional[Callable[[TranscriptResult], None]] = None,
    ):
        """Initialize Google STT streaming client.

        Args:
            language_code: Language code (e.g., "en-US")
            sample_rate: Input audio sample rate (will resample if different)
            single_utterance: Stop after first complete utterance
            interim_results: Enable interim (partial) results
            on_transcript: Callback for transcript results
        """
        self.client = speech.SpeechClient()
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.single_utterance = single_utterance
        self.interim_results = interim_results
        self.on_transcript = on_transcript

        # Streaming state
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._result_queue: asyncio.Queue[TranscriptResult] = asyncio.Queue()
        self._is_streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Build recognition config
        self._config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="latest_short",  # Optimized for short utterances
        )

        self._streaming_config = speech.StreamingRecognitionConfig(
            config=self._config,
            interim_results=interim_results,
            single_utterance=single_utterance,
        )

        logger.info(
            "GoogleSTTStream initialized: lang=%s, rate=%dHz, single_utterance=%s",
            language_code, sample_rate, single_utterance
        )

    def start_stream(self) -> None:
        """Start the streaming recognition session."""
        if self._is_streaming:
            logger.warning("Stream already active, ignoring start_stream()")
            return

        self._stop_event.clear()
        self._audio_queue = queue.Queue()
        self._is_streaming = True

        # Start streaming thread
        self._stream_thread = threading.Thread(
            target=self._streaming_worker,
            daemon=True
        )
        self._stream_thread.start()
        logger.debug("STT streaming started")

    def stop_stream(self) -> None:
        """Stop the streaming recognition session."""
        if not self._is_streaming:
            return

        self._is_streaming = False
        self._stop_event.set()

        # Signal end of audio
        self._audio_queue.put(None)

        # Wait for thread to finish
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)

        logger.debug("STT streaming stopped")

    def feed_audio(
        self,
        audio: NDArray[np.int16],
        input_sample_rate: Optional[int] = None
    ) -> None:
        """Feed audio data to the STT stream.

        Args:
            audio: Audio samples as int16 numpy array
            input_sample_rate: Sample rate of input (resamples if different)
        """
        if not self._is_streaming:
            return

        # Resample if needed
        if input_sample_rate and input_sample_rate != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / input_sample_rate)
            audio = resample(audio.astype(np.float32), num_samples).astype(np.int16)

        # Convert to bytes and queue
        audio_bytes = audio.tobytes()
        self._audio_queue.put(audio_bytes)

    def _audio_generator(self):
        """Generator that yields audio chunks for streaming."""
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def _streaming_worker(self) -> None:
        """Background worker that handles the streaming recognition."""
        try:
            requests = self._audio_generator()
            responses = self.client.streaming_recognize(
                self._streaming_config, requests
            )

            for response in responses:
                if self._stop_event.is_set():
                    break

                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript_result = TranscriptResult(
                    text=alternative.transcript,
                    is_final=result.is_final,
                    confidence=alternative.confidence if result.is_final else 0.0,
                    stability=result.stability if hasattr(result, 'stability') else 0.0,
                )

                # Queue result for async consumption
                try:
                    self._result_queue.put_nowait(transcript_result)
                except asyncio.QueueFull:
                    pass

                # Call sync callback if provided
                if self.on_transcript:
                    try:
                        self.on_transcript(transcript_result)
                    except Exception as e:
                        logger.error("Transcript callback error: %s", e)

                # Log results
                if result.is_final:
                    logger.info("Final transcript: '%s' (conf=%.2f)",
                               alternative.transcript, alternative.confidence)
                else:
                    logger.debug("Interim transcript: '%s'", alternative.transcript)

                # If single_utterance and final result, we're done
                if self.single_utterance and result.is_final:
                    break

        except Exception as e:
            if "iterating" not in str(e).lower():
                logger.error("STT streaming error: %s", e)
        finally:
            self._is_streaming = False

    async def get_transcript(self, timeout: float = 0.1) -> Optional[TranscriptResult]:
        """Get the next transcript result (async).

        Args:
            timeout: How long to wait for a result

        Returns:
            TranscriptResult or None if no result available
        """
        try:
            return await asyncio.wait_for(
                self._result_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def get_all_transcripts(self) -> AsyncIterator[TranscriptResult]:
        """Async iterator over all transcript results."""
        while self._is_streaming or not self._result_queue.empty():
            result = await self.get_transcript(timeout=0.1)
            if result:
                yield result

    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming

    def clear_queue(self) -> None:
        """Clear any pending results."""
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @staticmethod
    def get_supported_sample_rate() -> int:
        """Return the preferred sample rate for Google STT."""
        return GOOGLE_SAMPLE_RATE


class SimpleGoogleSTT:
    """Simple non-streaming Google STT for one-shot recognition.

    Useful for testing or when streaming isn't needed.
    """

    def __init__(self, language_code: str = "en-US"):
        """Initialize simple STT client.

        Args:
            language_code: Language code (e.g., "en-US")
        """
        self.client = speech.SpeechClient()
        self.language_code = language_code

    def transcribe(
        self,
        audio: NDArray[np.int16],
        sample_rate: int = GOOGLE_SAMPLE_RATE
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio samples as int16 numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=True,
        )

        audio_content = speech.RecognitionAudio(content=audio.tobytes())

        try:
            response = self.client.recognize(config=config, audio=audio_content)

            if response.results:
                return response.results[0].alternatives[0].transcript
            return ""

        except Exception as e:
            logger.error("STT transcription failed: %s", e)
            raise
