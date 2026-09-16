//===- ChatTemplate.h -----------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Chat template for formatting multi-turn conversations.
//
// Loads a JSON config that defines how role markers and special tokens are
// arranged, then applies it to a message list to produce the prompt string
// expected by the model.
//
//===----------------------------------------------------------------------===//

#ifndef FRONTEND_INTERFACES_BUDDY_LLM_CHATTEMPLATE
#define FRONTEND_INTERFACES_BUDDY_LLM_CHATTEMPLATE

#include <jsoncons/json.hpp>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {

struct Message {
  std::string role; // "system", "user", "assistant"
  std::string content;
};

class ChatTemplate {
public:
  /// Load a chat template from a JSON config file.
  /// Throws std::runtime_error on failure.
  static ChatTemplate fromFile(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("Failed to open chat template file: " + path);
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());

    jsoncons::json root;
    try {
      root = jsoncons::json::parse(text);
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to parse chat template JSON: " +
                               std::string(e.what()));
    }
    return fromJSON(root);
  }

  /// Apply the template to a message list, producing the full prompt string.
  std::string apply(const std::vector<Message> &messages) const {
    std::string result;

    if (addBos_) {
      result += bosToken_;
    }

    std::string pendingSystem;

    for (const auto &msg : messages) {
      // If systemInFirstUser is set, defer the system message content.
      if (msg.role == "system" && systemInFirstUser_) {
        pendingSystem = msg.content;
        continue;
      }

      auto it = rolePrefixes_.find(msg.role);
      if (it != rolePrefixes_.end()) {
        result += it->second;
      }

      // Prepend deferred system content into the first user message.
      if (msg.role == "user" && !pendingSystem.empty()) {
        result += pendingSystem;
        result += "\n\n";
        pendingSystem.clear();
      }

      result += msg.content;
      result += roleSuffix_;
      result += turnSuffix_;
    }

    if (addGenerationPrompt_) {
      auto it = rolePrefixes_.find("assistant");
      if (it != rolePrefixes_.end()) {
        result += it->second;
      }
    }

    return result;
  }

  const std::vector<int> &stopTokenIds() const { return stopTokenIds_; }
  const std::vector<std::string> &stopTokens() const { return stopTokens_; }
  const std::string &bosToken() const { return bosToken_; }
  const std::string &eosToken() const { return eosToken_; }

private:
  std::string bosToken_;
  std::string eosToken_;
  std::map<std::string, std::string> rolePrefixes_;
  std::string roleSuffix_;
  std::string turnSuffix_;
  std::vector<std::string> stopTokens_;
  std::vector<int> stopTokenIds_;
  bool addBos_ = true;
  bool addGenerationPrompt_ = true;
  bool systemInFirstUser_ = false;

  static std::string getStr(const jsoncons::json &obj, const std::string &key,
                            const std::string &defaultVal = "") {
    if (obj.contains(key) && obj[key].is_string())
      return obj[key].as<std::string>();
    return defaultVal;
  }

  static bool getBool(const jsoncons::json &obj, const std::string &key,
                      bool defaultVal = false) {
    if (obj.contains(key) && obj[key].is_bool())
      return obj[key].as<bool>();
    return defaultVal;
  }

  static ChatTemplate fromJSON(const jsoncons::json &root) {
    ChatTemplate tmpl;
    if (!root.is_object()) {
      throw std::runtime_error("Chat template JSON root must be an object");
    }

    tmpl.bosToken_ = getStr(root, "bos_token");
    tmpl.eosToken_ = getStr(root, "eos_token");
    tmpl.roleSuffix_ = getStr(root, "role_suffix");
    tmpl.turnSuffix_ = getStr(root, "turn_suffix");
    tmpl.addBos_ = getBool(root, "add_bos", true);
    tmpl.addGenerationPrompt_ = getBool(root, "add_generation_prompt", true);
    tmpl.systemInFirstUser_ = getBool(root, "system_in_first_user", false);

    if (root.contains("roles") && root["roles"].is_object()) {
      for (const auto &member : root["roles"].object_range()) {
        if (member.value().is_string())
          tmpl.rolePrefixes_[std::string(member.key())] =
              member.value().as<std::string>();
      }
    }

    if (root.contains("stop_tokens") && root["stop_tokens"].is_array()) {
      for (const auto &v : root["stop_tokens"].array_range())
        if (v.is_string())
          tmpl.stopTokens_.push_back(v.as<std::string>());
    }

    if (root.contains("stop_token_ids") && root["stop_token_ids"].is_array()) {
      for (const auto &v : root["stop_token_ids"].array_range())
        if (v.is_number())
          tmpl.stopTokenIds_.push_back(static_cast<int>(v.as<int64_t>()));
    }

    return tmpl;
  }
};

} // namespace buddy

#endif // FRONTEND_INTERFACES_BUDDY_LLM_CHATTEMPLATE
