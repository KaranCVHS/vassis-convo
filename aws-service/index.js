/**
 * index.js
 * Entry point for a custom Speech-to-Speech streaming application.
 * This server handles real-time audio streaming between clients and a custom audio bot,
 * managing the WebSocket communication and data flow.
 * 
 * @author Agent Voice Response <info@agentvoiceresponse.com>
 * @see [https://www.agentvoiceresponse.com](https://www.agentvoiceresponse.com)
 */

const express = require("express");
const WebSocket = require("ws");
require("dotenv").config();

// Initialize Express application
const app = express();

/**
 * Creates and configures a WebSocket connection to the custom audio bot.
 * 
 * @returns {WebSocket} Configured WebSocket instance
 */
const connectToCustomBot = () => {
  // --- Make sure this is the correct IP address for your Python server.
  // --- If the Python script is running on the SAME machine, use 'ws://localhost:8765'
  const botUrl = "ws://0.0.0.0:8766";
  console.log(`Connecting to custom audio bot at ${botUrl}`);
  return new WebSocket(botUrl);
};

/**
 * Stream Processing
 */

/**
 * Handles incoming client audio stream and manages communication with the custom audio bot.
 * Implements buffering for audio chunks received before the WebSocket connection is established.
 * 
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
const handleAudioStream = async (req, res) => {
  console.log("New audio stream received");

  const bufferedChunks = [];
  let isWsConnected = false;
  const ws = connectToCustomBot();

  // Configure WebSocket event handlers
  ws.on("open", () => {
    console.log("WebSocket connected to custom audio bot");
    isWsConnected = true;

    // <<< ADDED FOR DEBUGGING
    console.log(`Processing ${bufferedChunks.length} buffered chunk(s).`);

    // Process any buffered audio chunks
    bufferedChunks.forEach((chunk) => {
      // <<< ADDED FOR DEBUGGING
      console.log(`Forwarding buffered chunk of size ${chunk.length} to bot.`);
      ws.send(chunk); // Send raw binary data
    });
    // Clear the buffer after sending
    bufferedChunks.length = 0;
  });

  ws.on("message", (data) => {
    // <<< ADDED FOR DEBUGGING
    console.log(`Received message of size ${data.length} from bot. Forwarding to client.`);
    // The 'data' is the raw binary audio buffer from the bot.
    // We forward it directly to the client.
    res.write(data);
  });

  ws.on("close", (code, reason) => {
    // Use toString() on reason buffer for better readability
    console.log(`WebSocket connection closed. Code: ${code}, Reason: ${reason.toString()}`);
    isWsConnected = false;
    // Ensure the client response stream is closed when the bot connection ends.
    if (!res.writableEnded) {
      res.end();
    }
  });

  ws.on("error", (err) => {
    console.error("WebSocket error:", err);
    isWsConnected = false;
    // Clean up and notify client on error
    if (!res.writableEnded) {
      try {
        res.status(500).json({ message: "WebSocket connection error" });
      } catch (error) {
        console.error("Failed to send error response to client:", error);
      }
    }
  });

  // Handle incoming audio data from the client
  req.on("data", (chunk) => {
    // <<< ADDED FOR DEBUGGING
    console.log(`Received chunk of size ${chunk.length} from client.`);
    try {
      if (isWsConnected) {
        // <<< ADDED FOR DEBUGGING
        console.log("-> Forwarding chunk directly to bot.");
        // If connected, send the raw audio chunk directly.
        ws.send(chunk);
      } else {
        // <<< ADDED FOR DEBUGGING
        console.log("-> Buffering chunk (bot not connected yet).");
        // If not yet connected, buffer the chunk.
        bufferedChunks.push(chunk);
      }
    } catch (error) {
      console.error("Error processing audio chunk:", error);
    }
  });

  // Handle the end of the client's request stream
  req.on("end", () => {
    console.log("Request stream from client ended");
    // Close the WebSocket connection to the bot when the client is done sending audio.
    // Note: Some bot protocols may require a specific "end-of-stream" message
    // instead of just closing the connection. Adjust if needed.
    if (ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
    if (!res.writableEnded) {
      res.end();
    }
  });

  // Handle errors in the client's request stream
  req.on("error", (err) => {
    console.error("Request error:", err);
    try {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
      if (!res.writableEnded) {
        res.status(500).json({ message: "Stream error" });
      }
    } catch (error) {
      console.error("Error closing WebSocket on request error:", error);
    }
  });

  // Set required headers for streaming binary audio data
  res.setHeader("Content-Type", "application/octet-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
};

// API Endpoints
app.post("/speech-to-speech-stream", handleAudioStream);

// Start server
const PORT = process.env.PORT || 6030;
app.listen(PORT, () => {
  console.log(`Custom Speech-to-Speech server running on port ${PORT}`);
});