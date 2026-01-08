"""Claude-based conversation handler for Reachy Mini.

Replaces OpenAI Realtime with Claude API + Google Cloud STT/TTS.
"""

import json
import base64
import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from anthropic import Anthropic
from fastrtc import AdditionalOutputs, AsyncStreamHandler, audio_to_int16
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.google_stt import GoogleSTTStream, TranscriptResult, GOOGLE_SAMPLE_RATE
from reachy_mini_conversation_app.google_tts import GoogleTTS, OUTPUT_SAMPLE_RATE
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)

logger = logging.getLogger(__name__)

# Audio configuration
HANDLER_SAMPLE_RATE = 24000  # Match Reachy Mini's preferred rate
AUDIO_CHUNK_SIZE = 4800  # 200ms at 24kHz


def convert_tools_to_claude_format(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tool specs to Claude format.

    OpenAI format: {"type": "function", "name": ..., "description": ..., "parameters": ...}
    Claude format: {"name": ..., "description": ..., "input_schema": ...}
    """
    claude_tools = []
    for spec in tool_specs:
        claude_tool = {
            "name": spec.get("name"),
            "description": spec.get("description"),
            "input_schema": spec.get("parameters", {"type": "object", "properties": {}}),
        }
        claude_tools.append(claude_tool)
    return claude_tools


class ClaudeConversationHandler(AsyncStreamHandler):
    """Claude-based conversation handler using Google STT/TTS.

    This handler:
    1. Receives audio from microphone via receive()
    2. Streams to Google STT for transcription
    3. Sends transcripts to Claude API with tool support
    4. Synthesizes responses via Google TTS
    5. Returns audio via emit()
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the Claude conversation handler.

        Args:
            deps: Tool dependencies (robot, movement manager, etc.)
            gradio_mode: Whether running with Gradio UI
            instance_path: Path to instance directory
            model: Claude model to use
        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=HANDLER_SAMPLE_RATE,
            input_sample_rate=HANDLER_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self.model = model

        # Claude client (initialized in start_up)
        self.client: Optional[Anthropic] = None

        # Conversation history
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt = ""

        # STT/TTS components (initialized in start_up)
        self.stt: Optional[GoogleSTTStream] = None
        self.tts: Optional[GoogleTTS] = None

        # Audio queues
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        self._audio_buffer: List[NDArray[np.int16]] = []
        self._current_audio_index = 0

        # State management
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self._shutdown_requested = False

        # Timing
        self.last_activity_time = 0.0
        self.start_time = 0.0

        # Current transcript accumulator
        self._current_transcript = ""
        self._transcript_lock = asyncio.Lock()

        # Event loop reference (set in start_up)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info("ClaudeConversationHandler initialized with model=%s", model)

    def copy(self) -> "ClaudeConversationHandler":
        """Create a copy of the handler (required by fastrtc)."""
        return ClaudeConversationHandler(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            self.model,
        )

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime.

        Updates the global config's selected profile for subsequent calls.
        Reloads the system prompt for the next conversation turn.

        Returns a short status message for UI feedback.
        """
        try:
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile
            from reachy_mini_conversation_app.prompts import get_session_instructions

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            # Reload the system prompt
            try:
                self.system_prompt = get_session_instructions()
                # Clear conversation history to start fresh with new personality
                self.messages = []
                logger.info("Reloaded system prompt for profile %r", profile)
            except Exception as e:
                logger.error("Failed to reload system prompt: %s", e)
                return f"Failed to apply personality: {e}"

            display_name = profile or "(built-in default)"
            return f"Applied personality: {display_name}"

        except Exception as e:
            logger.error("apply_personality failed: %s", e)
            return f"Error: {e}"

    async def get_available_voices(self) -> list[str]:
        """Return available voices for Claude mode (Google TTS voices).

        For now returns a simple list since we use Google TTS.
        """
        # Google TTS Standard voices that work well
        return [
            "en-US-Standard-F",
            "en-US-Standard-A",
            "en-US-Standard-B",
            "en-US-Standard-C",
            "en-US-Standard-D",
            "en-US-Standard-E",
            "en-US-Standard-G",
            "en-US-Standard-H",
            "en-US-Standard-I",
            "en-US-Standard-J",
        ]

    async def start_up(self) -> None:
        """Initialize the handler components."""
        logger.info("Starting Claude conversation handler...")

        # Save reference to event loop for cross-thread callbacks
        self._loop = asyncio.get_running_loop()

        # Get API key
        api_key = getattr(config, 'ANTHROPIC_API_KEY', None)
        if not api_key:
            import os
            api_key = os.getenv('ANTHROPIC_API_KEY')

        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set!")
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

        # Initialize Claude client
        self.client = Anthropic(api_key=api_key)

        # Initialize Google STT
        stt_language = getattr(config, 'GOOGLE_STT_LANGUAGE', 'en-US')
        self.stt = GoogleSTTStream(
            language_code=stt_language,
            sample_rate=GOOGLE_SAMPLE_RATE,
            single_utterance=True,
            interim_results=True,
            on_transcript=self._on_transcript_sync,
        )

        # Initialize Google TTS
        tts_voice = getattr(config, 'GOOGLE_TTS_VOICE', 'en-US-Neural2-F')
        tts_language = getattr(config, 'GOOGLE_TTS_LANGUAGE', 'en-US')
        self.tts = GoogleTTS(
            voice_name=tts_voice,
            language_code=tts_language,
        )

        # Load system prompt
        self.system_prompt = get_session_instructions()

        # Set timing
        loop = asyncio.get_event_loop()
        self.start_time = loop.time()
        self.last_activity_time = loop.time()

        logger.info("Claude handler ready: model=%s, stt=%s, tts=%s",
                   self.model, stt_language, tts_voice)

    def _on_transcript_sync(self, result: TranscriptResult) -> None:
        """Synchronous callback for STT results (called from STT thread)."""
        logger.info("_on_transcript_sync called: is_final=%s, text='%s'", result.is_final, result.text[:50] if result.text else "")
        # Schedule async handling on the saved event loop
        if self._loop is None:
            logger.error("Event loop not set - start_up not called?")
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._handle_transcript(result),
                self._loop
            )
            logger.debug("Scheduled _handle_transcript on main loop")
        except Exception as e:
            logger.error("Could not schedule transcript handling: %s", e)

    async def _handle_transcript(self, result: TranscriptResult) -> None:
        """Handle a transcript result from STT."""
        async with self._transcript_lock:
            if result.is_final:
                # Final transcript - process with Claude
                self._current_transcript = result.text
                logger.info("User said: '%s'", result.text)

                # Emit user transcript to UI
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": result.text})
                )

                # Stop listening, start processing
                self.is_listening = False
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)

                # Process with Claude
                asyncio.create_task(self._process_with_claude(result.text))

            else:
                # Interim transcript - emit for UI feedback
                self._current_transcript = result.text
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user_partial", "content": result.text})
                )

    async def _process_with_claude(self, user_text: str) -> None:
        """Send user text to Claude and handle response."""
        if not self.client:
            logger.error("Claude client not initialized")
            return

        self.is_processing = True
        self.last_activity_time = asyncio.get_event_loop().time()

        try:
            # Add user message to history
            self.messages.append({
                "role": "user",
                "content": user_text,
            })

            # Get tool specs in Claude format
            tool_specs = get_tool_specs()
            claude_tools = convert_tools_to_claude_format(tool_specs)

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=self.messages,
                tools=claude_tools if claude_tools else None,
            )

            # Process response
            assistant_text = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    assistant_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # Handle tool calls
            if tool_calls:
                await self._handle_tool_calls(tool_calls, response)
            elif assistant_text:
                # Regular text response - synthesize and play
                await self._speak_response(assistant_text)

                # Add to history
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_text,
                })

        except Exception as e:
            logger.error("Claude API error: %s", e)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[Error: {e}]"})
            )

        finally:
            self.is_processing = False

    async def _handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        initial_response: Any
    ) -> None:
        """Handle tool calls from Claude's response."""
        # Add assistant message with tool use to history
        self.messages.append({
            "role": "assistant",
            "content": initial_response.content,
        })

        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            call_id = tool_call["id"]

            logger.info("Executing tool: %s", tool_name)

            # Emit tool call notification
            await self.output_queue.put(
                AdditionalOutputs({
                    "role": "assistant",
                    "content": f"Using tool: {tool_name}",
                    "metadata": {"title": f"Tool: {tool_name}", "status": "pending"},
                })
            )

            try:
                # Dispatch tool call using existing system
                result = await dispatch_tool_call(
                    tool_name,
                    json.dumps(tool_input),
                    self.deps
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": json.dumps(result),
                })

                # Emit tool result
                await self.output_queue.put(
                    AdditionalOutputs({
                        "role": "assistant",
                        "content": json.dumps(result),
                        "metadata": {"title": f"Tool: {tool_name}", "status": "done"},
                    })
                )

            except Exception as e:
                logger.error("Tool execution failed: %s", e)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": json.dumps({"error": str(e)}),
                    "is_error": True,
                })

        # Add tool results to messages
        self.messages.append({
            "role": "user",
            "content": tool_results,
        })

        # Get follow-up response from Claude
        tool_specs = get_tool_specs()
        claude_tools = convert_tools_to_claude_format(tool_specs)

        follow_up = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            messages=self.messages,
            tools=claude_tools if claude_tools else None,
        )

        # Extract text from follow-up
        follow_up_text = ""
        for block in follow_up.content:
            if block.type == "text":
                follow_up_text += block.text

        if follow_up_text:
            await self._speak_response(follow_up_text)
            self.messages.append({
                "role": "assistant",
                "content": follow_up_text,
            })

    async def _speak_response(self, text: str) -> None:
        """Synthesize and queue audio for response text."""
        if not text or not self.tts:
            logger.warning("_speak_response: no text or TTS not initialized")
            return

        logger.info("Speaking response: '%s...' (%d chars)", text[:50], len(text))
        self.is_speaking = True

        try:
            # Emit transcript
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": text})
            )

            # Synthesize audio
            logger.info("Synthesizing TTS...")
            audio = self.tts.synthesize(text)
            logger.info("TTS synthesized %d samples", len(audio))

            if len(audio) == 0:
                return

            # Feed to head wobbler if available
            if self.deps.head_wobbler is not None:
                # Encode as base64 for head wobbler (matches OpenAI format)
                audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")
                self.deps.head_wobbler.feed(audio_b64)

            # Play audio directly through PipeWire (routes to default sink - Scarlett)
            try:
                import subprocess
                import tempfile
                import wave

                # Write audio to a temp WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                    with wave.open(f, 'wb') as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)  # 16-bit
                        wav.setframerate(OUTPUT_SAMPLE_RATE)
                        wav.writeframes(audio.tobytes())

                logger.info("Playing audio via pw-play (PipeWire -> Scarlett)...")
                # Use pw-play which respects PipeWire default sink
                proc = subprocess.Popen(
                    ['pw-play', temp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                proc.wait()
                logger.info("Audio playback complete")

                # Clean up temp file
                import os
                os.unlink(temp_path)

            except Exception as play_err:
                logger.error("Direct audio playback failed: %s", play_err)
                # Fallback: queue for WebRTC (may not work if session closed)
                chunk_samples = AUDIO_CHUNK_SIZE
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i:i + chunk_samples]
                    chunk_2d = chunk.reshape(1, -1)
                    await self.output_queue.put((OUTPUT_SAMPLE_RATE, chunk_2d))
                logger.debug("Queued %d samples of TTS audio (fallback)", len(audio))

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)

        finally:
            self.is_speaking = False

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from microphone.

        Args:
            frame: Tuple of (sample_rate, audio_data)
        """
        if self._shutdown_requested:
            return

        input_sample_rate, audio_frame = frame

        # Debug: log audio frame info periodically
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        if self._frame_count % 100 == 1:  # Log every 100 frames
            max_amp = np.abs(audio_frame).max()
            logger.info(f"Audio frame #{self._frame_count}: rate={input_sample_rate}, len={len(audio_frame)}, max_amp={max_amp}")

        # Handle interruption - user starts speaking while robot is speaking
        if self.is_speaking and not self.is_listening:
            # Check if there's significant audio (simple VAD)
            if np.abs(audio_frame).max() > 500:  # Threshold for speech detection
                logger.debug("User interruption detected")
                # Clear output queue to stop current speech
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self.is_speaking = False

                # Reset head wobbler
                if self.deps.head_wobbler:
                    self.deps.head_wobbler.reset()

        # Start STT stream if not active
        if not self.is_listening and not self.is_processing:
            # Simple VAD - start listening on audio activity
            if np.abs(audio_frame).max() > 300:
                self.is_listening = True
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(True)

                if self.stt:
                    self.stt.start_stream()
                logger.debug("Started listening")

        # Feed audio to STT if listening
        if self.is_listening and self.stt:
            # Flatten if needed
            if audio_frame.ndim == 2:
                audio_frame = audio_frame.flatten()

            self.stt.feed_audio(audio_frame, input_sample_rate)

        self.last_activity_time = asyncio.get_event_loop().time()

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame or metadata to be played/displayed.

        Returns:
            Audio tuple, AdditionalOutputs, or None
        """
        # Handle idle behavior
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager and self.deps.movement_manager.is_idle():
            # Trigger idle behavior (breathing, looking around, etc.)
            self.last_activity_time = asyncio.get_event_loop().time()

        # Check for output
        try:
            item = await asyncio.wait_for(self.output_queue.get(), timeout=0.05)
            return item
        except asyncio.TimeoutError:
            return None

    async def shutdown(self) -> None:
        """Clean up resources."""
        self._shutdown_requested = True

        # Give STT a moment to finalize any pending transcript
        if self.stt and self.stt._is_streaming:
            logger.debug("Waiting for STT to finalize...")
            for _ in range(10):  # Wait up to 1 second
                await asyncio.sleep(0.1)
                # Check if we got a final transcript
                if not self.stt._is_streaming:
                    break

        # Stop STT
        if self.stt:
            self.stt.stop_stream()

        # Wait a bit for any scheduled callbacks to complete
        await asyncio.sleep(0.2)

        # Clear output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("Claude conversation handler shut down")

    def format_timestamp(self) -> str:
        """Format current timestamp."""
        elapsed = asyncio.get_event_loop().time() - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.messages.copy()
