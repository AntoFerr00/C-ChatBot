#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <curl/curl.h>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

// Use BlenderBot 400M distill with wait_for_model flag
const string API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill?wait_for_model=true";

// Function to load token from a file (e.g., "../token.txt" located outside the current folder)
string loadToken(const string& filePath) {
    ifstream tokenFile(filePath);
    if (!tokenFile.is_open()) {
        cerr << "Error: Could not open token file at " << filePath << endl;
        exit(EXIT_FAILURE);
    }
    stringstream buffer;
    buffer << tokenFile.rdbuf();
    string token = buffer.str();
    // Remove newline characters or extra whitespace
    token.erase(remove(token.begin(), token.end(), '\n'), token.end());
    token.erase(remove(token.begin(), token.end(), '\r'), token.end());
    return token;
}

// Callback function to store API response
size_t WriteCallback(void* contents, size_t size, size_t nmemb, string* output) {
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

// Function to call Hugging Face API and return the generated response
string getAIResponse(const string& input, const string& history, const string& token) {
    CURL* curl = curl_easy_init();
    if (!curl) return "CURL initialization failed";

    string response;
    // Combine conversation history with the new user prompt and bot marker.
    string formatted_input = history + "User: " + input + "\nBot:";
    
    // Build the JSON payload with additional generation parameters to improve coherence.
    json j;
    j["inputs"] = formatted_input;
    j["parameters"] = {
        {"truncation", "only_first"},
        {"max_new_tokens", 100},
        {"temperature", 0.7},
        {"top_p", 0.9},
        {"repetition_penalty", 1.2}
    };
    string payload = j.dump();

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + token).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, API_URL.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);

    // Retrieve HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        return "API request failed.";
    }
    
    if (http_code != 200) {
        return "API returned HTTP code " + to_string(http_code) + ". Response: " + response;
    }

    try {
        auto json_response = json::parse(response);
        if (json_response.is_array() && !json_response.empty()) {
            string full_response = json_response[0]["generated_text"].get<string>();
            // Remove the prompt only if it appears at the beginning.
            if (full_response.rfind(formatted_input, 0) == 0 && full_response.size() > formatted_input.size()) {
                return full_response.substr(formatted_input.size());
            } else {
                return full_response;
            }
        }
    } catch (json::parse_error& e) {
        return "JSON parsing error: " + string(e.what()) + ". Response: " + response;
    }

    return "No response generated.";
}

int main() {
    // Load the Hugging Face token from a file outside the current folder (adjust path if needed)
    string token = loadToken("../token.txt");

    cout << "AI-powered Chatbot (type 'exit' to quit)\n";

    string input, conversation_history = "";
    const int max_history_turns = 6;  // Limit conversation history to avoid overload

    vector<string> history_log;

    // Refined initial context to promote coherence and topic relevance.
    const string initial_context =
        "The following is a conversation with an AI assistant. "
        "The assistant is coherent, context-aware, and answers directly to the user's queries without deviating from the topic. "
        "Maintain the context and be specific in responses.\n";

    while (true) {
        cout << "\nYou: ";
        getline(cin, input);

        if (input == "exit" || input == "bye") {
            cout << "Chatbot: Goodbye!\n";
            break;
        }

        // Build conversation history including initial context.
        conversation_history = initial_context;
        int start = max(0, static_cast<int>(history_log.size()) - max_history_turns);
        for (int i = start; i < history_log.size(); i++) {
            conversation_history += history_log[i];
        }

        string response = getAIResponse(input, conversation_history, token);
        cout << "Chatbot:" << response << endl;

        // Update conversation history with the new messages.
        history_log.push_back("User: " + input + "\n");
        history_log.push_back("Bot: " + response + "\n");
    }
    
    return 0;
}
