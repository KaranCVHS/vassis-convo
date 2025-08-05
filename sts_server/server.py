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


from user_info import UserInfo
from salesforce_handler import SalesforceHandler

tools = [
    {
        "toolSpec": {
            "name": "verify_and_get_user_data",
            "description": "Verifies a user's identity using their phone number and a unique secret key. If verification is successful, this tool fetches the user's account and medication information. This tool MUST be called and return a successful verification before answering any questions about a user's personal data.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "description": "The user's 10-digit phone number, e.g., '5551234567'.",
                        },
                        "secret_key": {
                            "type": "string",
                            "description": "The user's secret key. It is the last three letters of their last name, the first letter of their first name, and their birth date as YYYYMMDD. Example: For John Smith born on Jan 3, 1990, the key is 'ithj19900103'.",
                        },
                    },
                    "required": ["phone", "secret_key"],
                }
            },
        }
    }
]

# --- Configuration ---
MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
WEBSOCKET_PORT = 8765

config = {
    "log_level": "debug",
    "region": AWS_REGION,
    "polly": {
        "Engine": "neural",
        "LanguageCode": "en-US",
        "VoiceId": "Joanna",
        "OutputFormat": "pcm",
        "SampleRate": "8000",
        "TextType": "ssml",
    },
    "vad": {"threshold": 0.5, "silence_sec": 1.5},
    # Removed 'polly_stream_end_delay_ms' as delays will now be handled by SSML breaks
}

# New constants for granular SSML generation, now used *inside* SSML
MAX_CHARS_PER_SSML_CHUNK = (
    80  # Max characters before forcing a split if no natural break
)
SHORT_BREAK_MS = 150  # Break after commas or forced splits
SENTENCE_BREAK_MS = (
    300  # Break after sentence-ending punctuation (increased for clarity)
)
LIST_ITEM_BREAK_MS = 400  # Break after list items (increased for clarity)


def _rollback_unmatched_tool_use(self):
    if self.messages_history and self.messages_history[-1]["role"] == "assistant":
        # Does the last assistant message contain any toolUse blocks?
        if any("toolUse" in c for c in self.messages_history[-1]["content"]):
            self.messages_history.pop()


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
    def __init__(self, app_state, salesforce_handler, user_session):
        self.app_state = app_state
        self.sf_handler = salesforce_handler
        self.user_session = user_session
        self.polly = boto3.client("polly", region_name=config["region"])
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", region_name=config["region"]
        )
        self.messages_history = []

    def _get_system_prompt(self):
        # Add comprehensive debugging
        printer(f"[DEBUG] User verified: {self.user_session.is_verified}", "info")
        printer(
            f"[DEBUG] SF data exists: {self.user_session.sf_data is not None}", "info"
        )
        printer(f"[DEBUG] SF data: {self.user_session.sf_data}", "info")
        printer(f"[DEBUG] First name: '{self.user_session.first_name}'", "info")
        printer(f"[DEBUG] Last name: '{self.user_session.last_name}'", "info")

        # Check current verification status and build dynamic prompt
        if self.user_session.is_verified and self.user_session.sf_data:
            printer(
                "[DEBUG] User is verified - generating VERIFIED system prompt",
                "info",
            )
            # User is verified - include their data in the system prompt
            verification_status = f"""**CURRENT USER STATUS: VERIFIED**
    - User: {self.user_session.first_name} {self.user_session.last_name}
    - You may answer all questions about their healthcare data using the information below.
    - DO NOT ask for verification again.

    **AVAILABLE USER DATA:**
    - Benefit Status: {self.user_session.sf_data.get("CoverageBenefit", {}).get("status")}
    - PA Status: {self.user_session.sf_data.get("CarePreauth", {}).get("outcome")}
    - PA Identifier: {self.user_session.sf_data.get("CarePreauth", {}).get("identifier")}
    - Medication Status: {self.user_session.sf_data.get("MedicationRequest", {}).get("status")}
    - Payment Status: {self.user_session.sf_data.get("Payments", {}).get("status")}
    - Payment Amount: ${self.user_session.sf_data.get("Payments", {}).get("price")}"""

            verification_instructions = """**IMPORTANT:** This user is already verified. Answer their healthcare questions directly using the data above."""
        else:
            printer(
                "[DEBUG] User is NOT verified - generating NOT VERIFIED system prompt",
                "info",
            )
            # User is not verified
            verification_status = """**CURRENT USER STATUS: NOT VERIFIED**
    - You MUST verify the user before answering any personal healthcare questions."""

            verification_instructions = """**VERIFICATION REQUIRED:**
    1. Ask for their **phone number and secret key together**.
    2. Use the `verify_and_get_user_data` tool to verify them.
    3. If verification fails, ask them to re-enter the information.
    4. Once verified, you can answer their questions."""

        system_prompt = f"""You are a helpful AI assistant for a healthcare provider.

    {verification_status}

    {verification_instructions}

    **USE CASES** (Once Verified)

    1. **Benefit Verification (BV)**
    - Tell patients whether their insurance covers a medication and what their copay or out-of-pocket cost is.
    - If "Covered" → Show copay and send secure payment link.
    - If "Not Covered" → Show full cost and send payment link.
    - If "Covered with Condition" → 
        - PA Pending → Notify it's in progress.
        - PA Approved → Show copay and send link.
        - PA Rejected → Show full cost and send link.
    - Fallback: "We are reviewing your benefit details. Please check back soon."

    2. **Prior Authorization (PA)**
    - Explain the PA process and current status.
    - If PA is "Initiated" or "Submitted" → Notify user it is pending.
    - If PA is "Approved" → Show copay and send secure payment link.
    - If PA is "Rejected" → Show out-of-pocket cost and send link.
    - If no PA record but medication needs one → Inform the user.
    - Fallback: "We are reviewing your benefit details. Please check back soon."

    3. **Payment (OOP)**
    - Clarify what the user owes and whether insurance covers the medication.
    - If Payment_status is "Unpaid" → Remind user of secure payment link.
    - Fallback: "We are reviewing your benefit details. Please check back soon."

    4. **Order Status**
    - Answer questions about shipment, delivery, and payment confirmation.
    - Based on Order_status:
        - Placed/Processing → Preparing for shipment.
        - Shipped → Tracking sent.
        - Out for delivery → On the way.
        - Delivered → Include delivery address.
        - Cancelled → Inform and suggest reordering.
    - If payment is pending → Inform order not placed yet.
    - Fallback: "We're reviewing your order details. Please check back soon."

    **REMINDERS**
    - Always use patient-friendly, clear language.
    - Never give raw data or medical advice.
    - Only operate within the verified patient support domain (coverage, PA, payment, order status).
    """

        # Log the final system prompt for debugging
        printer(
            f"[DEBUG] Generated system prompt length: {len(system_prompt)} chars",
            "info",
        )
        printer(f"[DEBUG] System prompt preview: {system_prompt[:200]}...", "info")

        return system_prompt

    def verify_and_get_user_data(self, phone, secret_key):
        """
        Verifies a user against Salesforce data using a phone number and a secret key.
        If successful, it populates the user_session with their data.
        This function is written defensively to handle cases where related records may not exist.
        """
        sf = self.sf_handler.get_session()
        if not sf:
            return json.dumps(
                {
                    "status": "False",
                    "verificationStatus": "False",
                    "error": "Salesforce connection not available.",
                }
            )

        try:
            # 1. Verify Account
            acct_query = (
                f"SELECT Id, Name, FirstName, LastName, PersonBirthdate "
                f"FROM Account WHERE Phone = '{phone}'"
            )
            acct_result = sf.query(acct_query)

            if acct_result["totalSize"] == 0:
                return json.dumps(
                    {
                        "status": "False",
                        "verificationStatus": "False",
                        "reason": "No account found with that phone number.",
                    }
                )

            account = acct_result["records"][0]
            first_name = account.get("FirstName")
            last_name = account.get("LastName")
            birthdate = account.get("PersonBirthdate")
            print(f"account: {account}")

            if not all([first_name, last_name, birthdate]):
                return json.dumps(
                    {
                        "status": "False",
                        "verificationStatus": "False",
                        "reason": "Account data is incomplete for verification.",
                    }
                )

            expected_key = (
                last_name[-3:] + first_name[0] + birthdate.replace("-", "")
            ).lower()
            secret_key_clean = secret_key.strip().lower()

            if secret_key_clean != expected_key:
                return json.dumps(
                    {
                        "status": "False",
                        "verificationStatus": "False",
                        "reason": "The provided secret key is incorrect.",
                    }
                )

            # --- Verification Successful ---
            self.user_session.is_verified = True
            self.user_session.first_name = first_name
            self.user_session.last_name = last_name

            # 2. Defensively Fetch Related Data
            account_id = account["Id"]
            print(f"account_id: {account_id}")

            # Initialize all data points with default "Not available" values
            coverage_outcome = "Not available"
            pa_identifier = "N/A"
            pa_outcome = "N/A"
            med_status = "Not available"
            payment_price = "N/A"
            payment_status = "N/A"

            # CareProgramEnrollee
            cpe_result = sf.query(
                f"SELECT Id FROM CareProgramEnrollee WHERE AccountId = '{account_id}'"
            )
            if cpe_result["totalSize"] > 0:
                cpe_id = cpe_result["records"][0].get("Id")

                # CoverageBenefit (only if enrollee exists)
                cb_result = sf.query(
                    f"SELECT Id, VHS_Outcome__c FROM CoverageBenefit WHERE VHS_Care_Program_Enrollee__c = '{cpe_id}' ORDER BY CreatedDate DESC LIMIT 1"
                )
                if cb_result["totalSize"] > 0:
                    coverage_outcome = cb_result["records"][0].get(
                        "VHS_Outcome__c", "Not available"
                    )
                    cb_id = cb_result["records"][0].get("Id")

                    # CarePreauth (only if coverage benefit exists)
                    pa_result = sf.query(
                        f"SELECT PreauthIdentifier, VHS_PAOutcome__c FROM CarePreauth WHERE VHS_Coverage_Benefit__c='{cb_id}' ORDER BY CreatedDate DESC LIMIT 1"
                    )
                    if pa_result["totalSize"] > 0:
                        pa_identifier = pa_result["records"][0].get("PreauthIdentifier")
                        pa_outcome = pa_result["records"][0].get("VHS_PAOutcome__c")

            # MedicationRequest
            med_result = sf.query(
                f"SELECT Id, Status FROM MedicationRequest WHERE PatientId = '{account_id}' ORDER BY CreatedDate DESC LIMIT 1"
            )
            if med_result["totalSize"] > 0:
                med_status = med_result["records"][0].get("Status", "Not available")
                med_id = med_result["records"][0].get("Id")

                # Payments (only if medication request exists)
                pay_result = sf.query(
                    f"SELECT Price__c, Status__c FROM Payments__c WHERE Medication_Request__c = '{med_id}' ORDER BY CreatedDate DESC LIMIT 1"
                )
                if pay_result["totalSize"] > 0:
                    payment_price = pay_result["records"][0].get("Price__c")
                    payment_status = pay_result["records"][0].get("Status__c")

            # 3. Store all fetched data in the user session object
            self.user_session.sf_data = {
                "CoverageBenefit": {"status": coverage_outcome},
                "CarePreauth": {"identifier": pa_identifier, "outcome": pa_outcome},
                "MedicationRequest": {"status": med_status},
                "Payments": {"price": payment_price, "status": payment_status},
            }

            # 4. Return success to the LLM
            return json.dumps(
                {
                    "status": "True",
                    "verificationStatus": "True",
                    "message": f"Successfully verified {first_name}. You can now answer their questions using the data provided.",
                    "data": self.user_session.sf_data,
                }
            )

        except Exception as e:
            print(f"[ERROR] Tool execution failed: {e}")
            return json.dumps(
                {
                    "status": "False",
                    "verificationStatus": "False",
                    "error": "An unexpected error occurred during data retrieval.",
                }
            )

    def processToolUse(self, toolName, toolInput):
        """Process tool use similar to your original processToolUse method"""
        tool = toolName.lower()
        printer(f"Tool Use - Name: {toolName}, Input: {toolInput}", "debug")

        try:
            if tool == "verify_and_get_user_data":
                printer("****Processing verify_and_get_user_data tool use****", "debug")

                phone = toolInput.get("phone", "")
                secret_key = toolInput.get("secret_key", "")

                if not phone or not secret_key:
                    return json.dumps(
                        {
                            "status": "False",
                            "verificationStatus": "False",
                            "error": "Phone number or secret key is missing. Please provide both.",
                        }
                    )

                result = self.verify_and_get_user_data(phone, secret_key)
                return result

            else:
                return json.dumps(
                    {"status": "False", "error": f"Tool {toolName} is not supported"}
                )

        except Exception as e:
            printer(f"Error in processToolUse: {str(e)}", "info")
            return json.dumps(
                {
                    "status": "False",
                    "error": f"An error occurred during tool execution: {str(e)}",
                }
            )

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

    async def _to_ssml_chunk_generator(self, bedrock_event_stream):
        """
        Async generator that yields SSML-formatted chunks of text from Bedrock text deltas.
        Adds SSML breaks for natural pauses and better interruptibility.
        """
        buffer = ""
        async for event in bedrock_event_stream:
            if self.app_state.was_interrupted():
                break
            if "contentBlockDelta" in event:
                text = event["contentBlockDelta"]["delta"]["text"]
                buffer += text

                # Regex to find sentence ends (., ?, ! followed by space), commas, or newlines
                # The parentheses around the delimiters make them part of the split result
                split_pattern = re.compile(r"([.?!]\s*|\s*,\s*|\n)")

                parts = split_pattern.split(buffer)

                # Process parts, attaching delimiters to the preceding text and yielding SSML chunks
                new_buffer = ""
                for i, part in enumerate(parts):
                    if part is None:
                        continue  # Skip None from potential empty groups

                    # If it's a delimiter
                    if split_pattern.fullmatch(part):
                        if new_buffer.strip():  # If there's text accumulated
                            if re.fullmatch(r"[.?!]\s*", part):  # Sentence end
                                yield f"<speak>{new_buffer.strip()}{part.strip()}<break time='{SENTENCE_BREAK_MS}ms'/></speak>"
                            elif re.fullmatch(r",\s*", part):  # Comma
                                yield f"<speak>{new_buffer.strip()}{part.strip()}<break time='{SHORT_BREAK_MS}ms'/></speak>"
                            elif re.fullmatch(r"\n", part):  # Newline/paragraph break
                                # Check if it's potentially a list item (starts with -)
                                if new_buffer.strip().startswith("-"):
                                    yield f"<speak>{new_buffer.strip()}<break time='{LIST_ITEM_BREAK_MS}ms'/></speak>"
                                else:
                                    yield f"<speak>{new_buffer.strip()}<break time='{SENTENCE_BREAK_MS}ms'/></speak>"  # Treat as sentence break for general newlines
                            new_buffer = ""  # Clear buffer after yielding
                        # If buffer is empty, it's a standalone delimiter. Only yield if it's a break.
                        elif re.fullmatch(r"\n", part):
                            # This handles cases where Bedrock might send only a newline
                            yield f"<speak><break time='{LIST_ITEM_BREAK_MS}ms'/></speak>"
                    else:  # It's a text segment
                        new_buffer += part
                        # Force split if the buffer gets too long without a natural break
                        if (
                            len(new_buffer) >= MAX_CHARS_PER_SSML_CHUNK
                            and " " in new_buffer
                        ):
                            last_space_idx = new_buffer.rfind(" ")
                            if last_space_idx != -1:
                                yield f"<speak>{new_buffer[:last_space_idx].strip()}<break time='{SHORT_BREAK_MS}ms'/></speak>"
                                new_buffer = new_buffer[
                                    last_space_idx:
                                ].strip()  # Keep trailing partial word
                            else:  # If no spaces, yield the whole thing if it's over limit
                                yield f"<speak>{new_buffer.strip()}<break time='{SHORT_BREAK_MS}ms'/></speak>"
                                new_buffer = ""

                buffer = new_buffer  # Update buffer with any remaining text

        # Yield any remaining text in the buffer as a final part
        if buffer.strip() and not self.app_state.was_interrupted():
            yield f"<speak>{buffer.strip()}</speak>"

    async def _send_audio_to_client(self, websocket, audio_data_stream):
        """Reads from Polly's audio stream in a non-blocking way and sends to client."""
        loop = asyncio.get_event_loop()
        try:
            while not self.app_state.was_interrupted():
                data = await loop.run_in_executor(None, audio_data_stream.read, 4096)
                if not data:
                    break
                await websocket.send(data)
                await asyncio.sleep(0.256)
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
        """Convert SSML text to speech and send to websocket"""
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

    def _rollback_messages(self, target_count):
        """Rollback message history to target count"""
        self.messages_history = self.messages_history[:target_count]
        printer(f"[DEBUG] Rolled back messages to count: {target_count}", "debug")

    def _add_user_message(self, text):
        """Add user message to history"""
        self.messages_history.append({"role": "user", "content": [{"text": text}]})

    def _add_assistant_message(self, text):
        """Add assistant message to history"""
        if text.strip():
            self.messages_history.append(
                {"role": "assistant", "content": [{"text": text.strip()}]}
            )

    async def _handle_tool_requests(self, tool_requests):
        """Process tool requests and return results for next turn"""
        tool_results = []

        for tool_request_content in tool_requests:
            tool_request = tool_request_content["toolUse"]
            tool_name = tool_request["name"]
            tool_id = tool_request["toolUseId"]
            tool_input = tool_request["input"]

            printer(f"[INFO] Tool: {tool_name}, Input: {tool_input}", "info")

            # Handle verification guardrail
            if (
                tool_name == "verify_and_get_user_data"
                and self.user_session.is_verified
            ):
                printer(
                    "[INFO] Guardrail: Blocked redundant verification call.", "info"
                )
                result_text = "User is already verified. Please answer the user's question directly using the information provided in the system prompt."
            else:
                # Execute the tool using our direct method
                result_text = self.processToolUse(tool_name, tool_input)

                # Try to format JSON nicely for logging
                try:
                    parsed_result = json.loads(result_text)
                    printer(
                        f"[INFO] Tool result: {json.dumps(parsed_result, indent=2)}",
                        "debug",
                    )
                except:
                    printer(f"[INFO] Tool result: {result_text}", "debug")

            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_id,
                        "content": [{"text": result_text}],
                    }
                }
            )
        return tool_results

    async def _process_streaming_response(self, websocket):
        """Handle streaming response from Bedrock after tool execution"""
        response_stream = self.bedrock_runtime.converse_stream(
            modelId=MODEL_ID,
            messages=self.messages_history,
            system=[{"text": self._get_system_prompt()}],
            toolConfig={"tools": tools},
        )

        bedrock_event_stream = self._stream_bedrock_events(response_stream["stream"])
        ssml_chunk_gen = self._to_ssml_chunk_generator(bedrock_event_stream)

        full_response = ""

        async for ssml_chunk in ssml_chunk_gen:
            if self.app_state.was_interrupted():
                break

            # Extract plain text for logging and history
            plain_text = re.sub(r"<[^>]+>", "", ssml_chunk).strip()
            if plain_text:
                print(f"ASSISTANT: {plain_text}", flush=True)
                full_response += plain_text + " "

            # Speak the SSML chunk
            await self._speak_text(websocket, ssml_chunk)

        return full_response.strip()

    async def _handle_simple_response(self, websocket, response_message):
        """Handle non-tool response from Bedrock"""
        text_content = ""
        for content in response_message.get("content", []):
            if "text" in content:
                text_content += content["text"]

        if text_content:
            ssml_response = f"<speak>{text_content}</speak>"
            await self._speak_text(websocket, ssml_response)
            print(f"ASSISTANT: {text_content}", flush=True)
            return text_content
        else:
            error_msg = "I'm sorry, I couldn't generate a response. Please try again."
            await self._speak_text(websocket, f"<speak>{error_msg}</speak>")
            return ""

    async def invoke_bedrock(self, websocket, text):
        """Main method to invoke Bedrock - simplified and easy to follow"""
        printer(f"\n[BEDROCK] Invoking with: '{text}'", "info")
        printer("[BEDROCK] Starting bot speech...", "debug")
        print("User info:", self.user_session.__dict__)

        # Start bot speech state
        self.app_state.start_bot_speech()

        # Save initial state for potential rollback
        initial_message_count = len(self.messages_history)

        try:
            # 1. Add user message to history
            self._add_user_message(text)

            # 2. Get initial response from Bedrock
            response = self.bedrock_runtime.converse(
                modelId=MODEL_ID,
                messages=self.messages_history,
                system=[{"text": self._get_system_prompt()}],
                toolConfig={"tools": tools},
            )

            response_message = response["output"]["message"]
            self.messages_history.append(response_message)

            # 3. Handle response based on type
            if response.get("stopReason") == "tool_use":
                # Extract and process tool requests
                tool_requests = [
                    c for c in response_message["content"] if "toolUse" in c
                ]

                if tool_requests:
                    # Execute tools and get results
                    tool_results = await self._handle_tool_requests(tool_requests)

                    # Add tool results to message history
                    self.messages_history.append(
                        {"role": "user", "content": tool_results}
                    )

                    # Get streaming response after tool execution
                    final_response = await self._process_streaming_response(websocket)

                    # Save the final response to history if successful
                    if final_response and not self.app_state.was_interrupted():
                        self._add_assistant_message(final_response)
                    else:
                        # Rollback on interruption or empty response
                        self._rollback_messages(initial_message_count)
                else:
                    # No tool results, rollback
                    self._rollback_messages(initial_message_count)
            else:
                # Handle simple text response
                final_response = await self._handle_simple_response(
                    websocket, response_message
                )

                if not final_response:
                    # Rollback if no valid response
                    self._rollback_messages(initial_message_count)

        except asyncio.CancelledError:
            printer("[BEDROCK] Bedrock task was cancelled by external signal.", "debug")
        except Exception as e:
            printer(f"[ERROR] Bedrock invocation error: {e}", "info")

            # Send error message if not interrupted
            if not self.app_state.was_interrupted():
                error_msg = "I'm sorry, I encountered an error. Please try again."
                await self._speak_text(websocket, f"<speak>{error_msg}</speak>")
        finally:
            print("", end="\n", flush=True)
            self.app_state.stop_bot_speech()


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
        try:
            self.sf_handler = SalesforceHandler()
        except Exception:
            printer(
                "[FATAL] Could not connect to Salesforce. Please check credentials.",
                "info",
            )
        self.websocket = websocket
        self.user_session = UserInfo()
        self.app_state = AppState()
        self.bedrock_wrapper = BedrockWrapper(
            self.app_state,
            salesforce_handler=self.sf_handler,
            user_session=self.user_session,
        )
        self.transcribe_client = TranscribeStreamingClient(region=config["region"])
        self.samplerate = int(config["polly"]["SampleRate"])
        self.VAD_CHUNK_SIZE_BYTES = 512
        self.audio_buffer = b""
        self.bot_response_task = None

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
            self.bot_response_task = asyncio.create_task(
                self.bedrock_wrapper.invoke_bedrock(self.websocket, final_transcript)
            )

    async def handle_connection(self):
        self._handle_transcription(
            "Hey there my phone number is 9388609635 and my secret key is nnyj20000318 help me verify"
        )

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
                        if self.bot_response_task and not self.bot_response_task.done():
                            self.bot_response_task.cancel()
                            self.app_state.interrupt()

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
                                await transcribe_stream.input_stream.end_stream()
                                await transcript_task
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

                    elif is_speech_chunk:
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
            if self.bot_response_task and not self.bot_response_task.done():
                self.bot_response_task.cancel()
            if transcript_task and not transcript_task.done():
                transcript_task.cancel()
            if transcribe_stream:
                try:
                    await transcribe_stream.input_stream.end_stream()
                except:
                    pass


async def main():
    print_startup_info()

    async def handler(websocket, path=None):
        printer(f"[CONNECTION] New client from {websocket.remote_address}", "info")
        ws_handler = WebSocketHandler(websocket)
        await ws_handler.handle_connection()

    async with websockets.serve(handler, "0.0.0.0", WEBSOCKET_PORT):
        printer(f"[INFO] Server started on ws://0.0.0.0:{WEBSOCKET_PORT}", "info")
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
