# AI-Powered Chatbot

This project is a C++ command-line chatbot that leverages Hugging Face's Inference API to generate responses using the [BlenderBot 400M Distill](https://huggingface.co/facebook/blenderbot-400M-distill) model. The chatbot maintains a conversation history, applies generation parameters to improve coherence, and loads the API token from an external file for security.

## Features

- **Conversational Interface:** Chat via the command line.
- **Coherent Responses:** Uses refined generation parameters (temperature, top-p, max new tokens, repetition penalty) for consistent output.
- **Conversation History:** Maintains a short history of the dialogue for context.
- **Secure Token Handling:** Loads the Hugging Face API token from an external file (e.g., `../token.txt`).

## Prerequisites

- **C++ Compiler:** Ensure you have a C++ compiler such as `g++` or `clang++` installed.
- **cURL Library:** The cURL library (and its development headers) is required for making HTTP requests.
- **nlohmann/json:** A header-only JSON library used for parsing JSON responses. You can download it from the [nlohmann/json GitHub repository](https://github.com/nlohmann/json).

### Installing Dependencies on Ubuntu

You can install the cURL development library using:

```bash
sudo apt-get update
sudo apt-get install libcurl4-openssl-dev
```

For the JSON library, you can either download the single header file (json.hpp) and place it in your include path or install it via a package manager if available.

## Compilation
To compile the chatbot code, use a command similar to the following:

```bash
g++ main.cpp -o chatbot -lcurl -lws2_32 -lwldap32 -lcrypt32
```

-std=c++11 sets the C++ standard.
-lcurl links the cURL library.
Replace main.cpp with the name of your source file if different.
Running the Chatbot
After compiling, run the executable:

```bash
./chatbot
```
The chatbot will start and display:

pgsql
```bash
AI-powered Chatbot (type 'exit' to quit)
```
Type your messages and see the responses. To exit the conversation, type exit or bye.

## Code Overview
main.cpp: Contains the main application logic.
Token Loading: The loadToken function reads the Hugging Face API token from an external file (e.g., ../token.txt).
HTTP Requests: Uses cURL to send POST requests to the Hugging Face Inference API.
JSON Parsing: Parses the API response using the nlohmann/json library.
Conversation Management: Maintains and limits conversation history to keep the context relevant.
Generation Parameters: Customizable parameters (like max_new_tokens, temperature, top_p, repetition_penalty) are sent with each request to control the response style.

## Customization
You can adjust the generation parameters in the getAIResponse function to fine-tune the model's behavior. For example:

max_new_tokens: Limits the length of responses.
temperature: Controls the randomness of output (lower values make the output more deterministic).
top_p: Implements nucleus sampling to filter low-probability tokens.
repetition_penalty: Reduces repetitive outputs.

## License
This project is provided as-is, without any warranty. Feel free to modify and distribute the code for personal use.

Acknowledgments
Hugging Face Inference API
BlenderBot 400M Distill
nlohmann/json
cURL
yaml


---

This README file explains the purpose of the project, its prerequisites, how to compile and run the code, and offers an overview of its components and customization options.