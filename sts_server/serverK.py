import asyncio
import json
import os
import time
import websockets
import boto3
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
import re
import torch
import numpy as np
import torchaudio
from dotenv import load_dotenv

load_dotenv()

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler

# --- Configuration ---
MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
WEBSOCKET_PORT = 8766
WORDS_PER_CHUNK = 9

config = {
    "log_level": "debug",
    "region": AWS_REGION,
    "polly": {
        "Engine": "neural",
        "LanguageCode": "en-US",
        "VoiceId": "Joanna",
        "OutputFormat": "pcm",
        "SampleRate": "8000",
        "TextType": "text",
    },
    "vad": {"threshold": 0.5, "silence_sec": 1.5},
    # Removed 'polly_stream_end_delay_ms' as delays will now be handled by SSML breaks
}

MAX_CHARS_PER_SSML_CHUNK = (
    45  # Max characters before forcing a split if no natural break
)
FORCED_BREAK_MS = 50  # Break after forced splits
SHORT_BREAK_MS = 100  # Break after commas
SENTENCE_BREAK_MS = 150  # Break after sentence-ending punctuation
LIST_ITEM_BREAK_MS = 250  # Break for list items or paragraphs (newlines)


# --- Centralized State Management ---
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.interrupt_event = threading.Event()
        self._is_bot_speaking = False

    def is_bot_speaking(self):
        with self.lock:
            return self._is_bot_speaking

    def start_bot_speech(self):
        with self.lock:
            self._is_bot_speaking = True
            self.interrupt_event.clear()
            printer("[STATE] Bot speech started.", "debug")

    def stop_bot_speech(self):
        with self.lock:
            self._is_bot_speaking = False
            self.interrupt_event.clear()
            printer("[STATE] Bot speech stopped.", "debug")

    def interrupt(self):
        if self.is_bot_speaking():
            printer("[STATE] Interrupt triggered!", "info")
            self.interrupt_event.set()

    def was_interrupted(self):
        return self.interrupt_event.is_set()


def printer(text, level):
    if config["log_level"] in ("info", "debug") and level == "info":
        print(text, flush=True)
    elif config["log_level"] == "debug" and level == "debug":
        print(text, flush=True)


class BedrockWrapper:
    def __init__(self, app_state):
        self.app_state = app_state
        self.polly = boto3.client("polly", region_name=config["region"])
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", region_name=config["region"]
        )
        self.messages_history = []

    def _get_system_prompt(self):
        return "You are a friendly and helpful voice assistant. Keep your responses concise and conversational."

    async def _stream_bedrock_events(self, stream):
        """Asynchronously yields events from the Bedrock stream without raising StopIteration into a future."""
        loop = asyncio.get_event_loop()
        stream_iterator = iter(stream)
        sentinel = object()

        while not self.app_state.was_interrupted():
            try:
                event = await loop.run_in_executor(
                    None, next, stream_iterator, sentinel
                )
                if event is sentinel:
                    break
                yield event
            except Exception as e:
                printer(f"[ERROR] Error in Bedrock stream iterator: {e}", "info")
                break

    # async def _to_word_chunk_generator(self, bedrock_event_stream):
    #     """
    #     Takes a stream of text deltas and yields SSML-formatted chunks of N words.
    #     """
    #     buffer = ""
    #     async for event in bedrock_event_stream:
    #         if self.app_state.was_interrupted():
    #             break
    #         if "contentBlockDelta" in event:
    #             text = event["contentBlockDelta"]["delta"]["text"]
    #             buffer += text

    #             words = buffer.split()

    #             if len(words) >= WORDS_PER_CHUNK:
    #                 chunk_to_yield = " ".join(words[:WORDS_PER_CHUNK])
    #                 # Wrap the chunk in SSML tags before yielding
    #                 yield f"<speak>{chunk_to_yield}</speak>"

    #                 buffer = " ".join(words[WORDS_PER_CHUNK:]) + " "

    #     # Also wrap the final remaining chunk in SSML tags
    #     if buffer.strip() and not self.app_state.was_interrupted():
    #         yield f"<speak>{buffer.strip()}</speak>"

    async def _to_ssml_chunk_generator(self, bedrock_event_stream):
        """
        Async generator that yields SSML-formatted chunks of text from Bedrock text deltas.
        It splits text at natural pauses (punctuation) to minimize latency.
        """
        buffer = ""
        # This regex splits the text by punctuation, but KEEPS the delimiter
        split_pattern = re.compile(r"([.?!,]\s*|\n)")

        async for event in bedrock_event_stream:
            if self.app_state.was_interrupted():
                break
            if "contentBlockDelta" in event:
                text = event["contentBlockDelta"]["delta"]["text"]
                buffer += text

                # Process the buffer as soon as it contains a delimiter
                parts = split_pattern.split(buffer)
                buffer = parts.pop()

                for i in range(0, len(parts), 2):
                    text_part = parts[i]
                    delimiter = parts[i + 1] if (i + 1) < len(parts) else ""

                    if not text_part and not delimiter:
                        continue

                    # Create the SSML with the correct break time based on the delimiter
                    if re.fullmatch(r"[.?!]\s*", delimiter):  # Sentence end
                        ssml_chunk = f"<speak>{text_part.strip()}{delimiter.strip()} <break time='{SENTENCE_BREAK_MS}ms'/></speak>"
                    elif re.fullmatch(r",\s*", delimiter):  # Comma
                        ssml_chunk = f"<speak>{text_part.strip()}{delimiter.strip()} <break time='{SHORT_BREAK_MS}ms'/></speak>"
                    elif re.fullmatch(
                        r"\n", delimiter
                    ):  # Newline (paragraph or list item)
                        break_time = (
                            LIST_ITEM_BREAK_MS
                            if text_part.strip().startswith("-")
                            else SENTENCE_BREAK_MS
                        )
                        ssml_chunk = f"<speak>{text_part.strip()} <break time='{break_time}ms'/></speak>"
                    else:  # No delimiter found yet, but buffer is long
                        ssml_chunk = f"<speak>{text_part.strip()}</speak>"  # Should not happen often due to logic below

                    yield ssml_chunk

                # SAFETY NET: If the buffer gets too long without any punctuation (long sentence),
                # force a split to ensure responsiveness.
                if len(buffer) > MAX_CHARS_PER_SSML_CHUNK:
                    last_space_idx = buffer.rfind(" ")
                    if last_space_idx != -1:
                        text_to_yield = buffer[:last_space_idx].strip()
                        yield f"<speak>{text_to_yield}<break time='{FORCED_BREAK_MS}ms'/></speak>"
                        buffer = buffer[last_space_idx:].strip()

        # Yield any final remaining text in the buffer after the stream ends
        if buffer.strip() and not self.app_state.was_interrupted():
            yield f"<speak>{buffer.strip()}</speak>"

    async def _synthesize_and_queue_audio(self, audio_queue, text_chunk):
        if not text_chunk.strip():
            return

        printer(f"[POLLY] Synthesizing: '{text_chunk}'", "info")
        try:
            polly_response = self.polly.synthesize_speech(
                Text=text_chunk,
                TextType="ssml",
                Engine=config["polly"]["Engine"],
                LanguageCode=config["polly"]["LanguageCode"],
                VoiceId=config["polly"]["VoiceId"],
                OutputFormat="pcm",
                SampleRate=config["polly"]["SampleRate"],
            )

            audio_stream = polly_response["AudioStream"]
            while True:
                chunk = audio_stream.read(4096)
                if not chunk:
                    break
                # The queue mechanism remains the same and is still essential
                await audio_queue.put(chunk)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            printer(
                f"[ERROR] Polly synthesis failed for chunk '{text_chunk}': {e}", "info"
            )

    async def _send_audio_to_client(self, websocket, audio_data_stream):
        """Reads from Polly's audio stream in a non-blocking way and sends to client."""
        loop = asyncio.get_event_loop()
        try:
            while not self.app_state.was_interrupted():
                data = await loop.run_in_executor(None, audio_data_stream.read, 4096)
                if not data:
                    break
                await websocket.send(data)
        except asyncio.CancelledError:
            printer("[AUDIO] Audio streaming cancelled.", "debug")
        except websockets.exceptions.ConnectionClosed:
            printer("[WARN] Client connection closed during audio streaming.", "info")
            self.app_state.interrupt()
        except Exception as e:
            printer(f"[ERROR] Error in audio streaming: {e}", "info")
        finally:
            if audio_data_stream:
                audio_data_stream.close()

    async def _speak_text(self, websocket, ssml_text):
        cleaned_ssml = re.sub(r"\*[^*]*\*", "", ssml_text)

        if not cleaned_ssml.strip() or re.fullmatch(
            r"\s*(<break[^>]*\/>\s*)*", cleaned_ssml
        ):
            return

        printer(f"[POLLY] Synthesizing: '{cleaned_ssml}'", "info")
        try:
            polly_response = self.polly.synthesize_speech(
                Text=cleaned_ssml,
                TextType=config["polly"]["TextType"],
                Engine=config["polly"]["Engine"],
                LanguageCode=config["polly"]["LanguageCode"],
                VoiceId=config["polly"]["VoiceId"],
                OutputFormat="pcm",
                SampleRate=config["polly"]["SampleRate"],
            )
            await self._send_audio_to_client(websocket, polly_response["AudioStream"])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            printer(f"[ERROR] Polly synthesis failed: {e}", "info")

    async def invoke_bedrock(self, audio_queue, text):
        printer(f"\n[BEDROCK] Invoking with: '{text}'", "info")
        self.app_state.start_bot_speech()
        full_assistant_response_raw = ""
        try:
            self.messages_history.append({"role": "user", "content": [{"text": text}]})
            response_stream = self.bedrock_runtime.converse_stream(
                modelId=MODEL_ID,
                messages=self.messages_history,
                system=[{"text": self._get_system_prompt()}],
            )

            bedrock_event_stream = self._stream_bedrock_events(
                response_stream["stream"]
            )
            ssml_chunk_gen = self._to_ssml_chunk_generator(bedrock_event_stream)

            async for ssml_chunk in ssml_chunk_gen:
                if self.app_state.was_interrupted():
                    printer("[BEDROCK] Interrupted, stopping text generation.", "debug")
                    break

                raw_text = re.sub("<[^<]+?>", "", ssml_chunk)
                full_assistant_response_raw += raw_text + " "
                print(f"ASSISTANT: {ssml_chunk}", end=" ", flush=True)

                await self._synthesize_and_queue_audio(audio_queue, ssml_chunk)

            if full_assistant_response_raw and not self.app_state.was_interrupted():
                self.messages_history.append(
                    {
                        "role": "assistant",
                        "content": [{"text": full_assistant_response_raw.strip()}],
                    }
                )
            else:
                if (
                    self.messages_history
                    and self.messages_history[-1]["role"] == "user"
                ):
                    self.messages_history.pop()

        except asyncio.CancelledError:
            printer("[BEDROCK] Bedrock task was cancelled by external signal.", "debug")
            if self.messages_history and self.messages_history[-1]["role"] == "user":
                self.messages_history.pop()
        except Exception as e:
            printer(f"[ERROR] Bedrock invocation error: {e}", "info")
            if (
                not self.app_state.was_interrupted()
                and self.messages_history
                and self.messages_history[-1]["role"] == "user"
            ):
                await self._synthesize_and_queue_audio(
                    audio_queue,
                    "I'm sorry, I encountered an error. Please try again.",
                )
                self.messages_history.pop()
        finally:
            print("", end="\n", flush=True)
            self.app_state.stop_bot_speech()
            await audio_queue.put(None)


class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, on_final_transcript):
        super().__init__(output_stream)
        self.full_transcript = []
        self.on_final_transcript = on_final_transcript

    async def handle_transcript_event(self, transcript_event):
        results = transcript_event.transcript.results
        if results and results[0].alternatives and not results[0].is_partial:
            transcript = results[0].alternatives[0].transcript
            print(f"USER: {transcript}", end=" ", flush=True)
            self.full_transcript.append(transcript)

    def get_full_transcript(self):
        full = " ".join(self.full_transcript).strip()
        if self.on_final_transcript and full:
            self.on_final_transcript(full)


class WebSocketHandler:
    def __init__(self, websocket):
        self.websocket = websocket
        self.app_state = AppState()
        self.bedrock_wrapper = BedrockWrapper(self.app_state)
        self.transcribe_client = TranscribeStreamingClient(region=config["region"])
        self.samplerate = int(config["polly"]["SampleRate"])
        self.VAD_CHUNK_SIZE_BYTES = 512
        self.audio_buffer = b""
        self.bot_response_task = None
        self.audio_sender_task = None
        self.audio_queue = None

        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.samplerate, new_freq=16000
            )
        except Exception as e:
            printer(f"[FATAL] Could not load Silero VAD model: {e}", "info")
            raise

    def _is_speech(self, chunk):
        if len(chunk) < self.VAD_CHUNK_SIZE_BYTES:
            return False
        try:
            audio_int16 = np.frombuffer(chunk, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32)
            audio_16k = self.resampler(audio_tensor)
            speech_prob = self.vad_model(audio_16k, 16000).item()
            is_speech_flag = speech_prob > config["vad"]["threshold"]
            # printer(f"[VAD] Prob: {speech_prob:.2f} -> {'SPEECH' if is_speech_flag else 'SILENCE'}", "debug")
            return is_speech_flag
        except Exception:
            return False

    def _handle_transcription(self, final_transcript):
        if final_transcript:
            self.audio_queue = asyncio.Queue()
            self.audio_sender_task = asyncio.create_task(self._audio_sender())
            self.bot_response_task = asyncio.create_task(
                self.bedrock_wrapper.invoke_bedrock(self.audio_queue, final_transcript)
            )

    async def _audio_sender(self):
        printer("[AUDIO_SENDER] Starting audio sender task.", "debug")
        try:
            while True:
                chunk = await self.audio_queue.get()

                if chunk is None:
                    printer(
                        "[AUDIO_SENDER] End of audio stream signal received.", "debug"
                    )
                    break

                await self.websocket.send(chunk)

        except asyncio.CancelledError:
            printer("[AUDIO_SENDER] Audio sender task cancelled.", "debug")
        except websockets.exceptions.ConnectionClosed:
            printer("[WARN] Client connection closed during audio sending.", "info")
            self.app_state.interrupt()  # Ensure other tasks know
        except Exception as e:
            printer(f"[ERROR] Error in audio sender: {e}", "info")
        finally:
            printer("[AUDIO_SENDER] Audio sender task finished.", "debug")

    def _clear_audio_queue(self):
        if not self.audio_queue:
            return
        printer("[QUEUE] Clearing audio queue.", "debug")
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        printer("[QUEUE] Audio queue cleared.", "debug")

    async def handle_connection(self):
        self._handle_transcription("User said hello.")

        is_transcribing = False
        transcribe_stream, transcript_handler, transcript_task = None, None, None
        silence_frames = 0

        printer("\n[INFO] Waiting for user to speak...", "info")
        try:
            async for raw_chunk in self.websocket:
                self.audio_buffer += raw_chunk
                while len(self.audio_buffer) >= self.VAD_CHUNK_SIZE_BYTES:
                    chunk_to_process = self.audio_buffer[: self.VAD_CHUNK_SIZE_BYTES]
                    self.audio_buffer = self.audio_buffer[self.VAD_CHUNK_SIZE_BYTES :]
                    is_speech_chunk = self._is_speech(chunk_to_process)

                    if self.app_state.is_bot_speaking() and is_speech_chunk:
                        printer("\n[INFO] Barge-in detected! Interrupting bot.", "info")
                        self.app_state.interrupt()
                        if self.bot_response_task and not self.bot_response_task.done():
                            self.bot_response_task.cancel()
                        if self.audio_sender_task and not self.audio_sender_task.done():
                            self.audio_sender_task.cancel()

                        self._clear_audio_queue()
                        self.app_state.stop_bot_speech()

                        if transcript_task and not transcript_task.done():
                            transcript_task.cancel()
                        if transcribe_stream:
                            await transcribe_stream.input_stream.end_stream()
                        (
                            is_transcribing,
                            transcribe_stream,
                            transcript_handler,
                            transcript_task,
                        ) = False, None, None, None

                    elif is_transcribing:
                        await transcribe_stream.input_stream.send_audio_event(
                            audio_chunk=chunk_to_process
                        )
                        if not is_speech_chunk:
                            silence_frames += 1
                            chunks_per_second = self.samplerate / (
                                self.VAD_CHUNK_SIZE_BYTES / 2
                            )
                            if silence_frames >= int(
                                chunks_per_second * config["vad"]["silence_sec"]
                            ):
                                printer("\n[INFO] End of speech detected.", "info")
                                if transcribe_stream:
                                    await transcribe_stream.input_stream.end_stream()
                                if transcript_task:
                                    await transcript_task
                                if transcript_handler:
                                    transcript_handler.get_full_transcript()
                                (
                                    is_transcribing,
                                    transcribe_stream,
                                    transcript_handler,
                                    transcript_task,
                                ) = False, None, None, None
                                printer("\n[INFO] Waiting for user to speak...", "info")
                        else:
                            silence_frames = 0

                    elif is_speech_chunk and not self.app_state.is_bot_speaking():
                        printer(
                            "\n[INFO] Speech detected, starting transcription...",
                            "info",
                        )
                        is_transcribing = True
                        silence_frames = 0
                        transcribe_stream = (
                            await self.transcribe_client.start_stream_transcription(
                                language_code="en-US",
                                media_sample_rate_hz=self.samplerate,
                                media_encoding="pcm",
                            )
                        )
                        transcript_handler = TranscriptHandler(
                            transcribe_stream.output_stream, self._handle_transcription
                        )
                        transcript_task = asyncio.create_task(
                            transcript_handler.handle_events()
                        )
                        await transcribe_stream.input_stream.send_audio_event(
                            audio_chunk=chunk_to_process
                        )

        except websockets.exceptions.ConnectionClosed as e:
            printer(f"[INFO] Client disconnected: {e.code}", "info")
        except Exception as e:
            printer(f"[FATAL] Connection handler error: {e}", "info")
        finally:
            printer("[CLEANUP] Cleaning up connection.", "debug")
            
            # Cancel the high-level tasks first
            if self.bot_response_task and not self.bot_response_task.done():
                self.bot_response_task.cancel()
            if self.audio_sender_task and not self.audio_sender_task.done():
                self.audio_sender_task.cancel()

            # Now handle the transcription stream shutdown gracefully
            if transcript_task and not transcript_task.done():
                try:
                    # Attempt to politely end the stream first.
                    # This might fail if the connection is already dead.
                    if transcribe_stream:
                        await transcribe_stream.input_stream.end_stream()
                except Exception as e:
                    printer(f"[CLEANUP] Error ending transcribe stream: {e}", "debug")
                finally:
                    # Now, forcefully cancel the task.
                    # This is guaranteed to stop it.
                    transcript_task.cancel()


async def main():
    print_startup_info()

    async def handler(websocket, path=None):
        printer(f"[CONNECTION] New client from {websocket.remote_address}", "info")
        ws_handler = WebSocketHandler(websocket)
        await ws_handler.handle_connection()

    async with websockets.serve(handler, "172.20.30.47", WEBSOCKET_PORT):
        printer(f"[INFO] Server started on ws://localhost:{WEBSOCKET_PORT}", "info")
        await asyncio.Future()


def print_startup_info():
    print("*" * 60)
    print(f"[INFO] Bedrock Model: {MODEL_ID}")
    print(f"[INFO] Polly Sample Rate: {config['polly']['SampleRate']}Hz")
    print("[INFO] Voice Assistant Server is ready.")
    print("*" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
