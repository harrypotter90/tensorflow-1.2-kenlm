/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef CTC_VOCABULARY_H
#define CTC_VOCABULARY_H

#include <fstream>
#include <istream>
#include <iostream>
#include <assert.h>
#include <unordered_map>
#include <sstream>

#include "utf8.h"

namespace tensorflow {
namespace ctc {

class Vocabulary {
public:
  ~Vocabulary() {
    delete char_list;
  }

  Vocabulary(const char *spec_file_path) {
    std::ifstream in(spec_file_path, std::ios::in);
    ReadFromFile(in);
    in.close();
  }

  Vocabulary(const wchar_t *char_list, int size) : size(size) {
    this->char_list = new wchar_t[size];
    for (int i = 0; i < size; i++) {
      this->char_list[i] = char_list[i];
      char_to_label[char_list[i]] = i;
    }
  }

  bool inline IsBlankLabel(int label) const {
    // If label is not contained in this vocabulary
    // it must be a blank label
    return label == size;
  }

  bool inline IsSpaceLabel(int label) const {
    return char_list[label] == ' ';
  }

  wchar_t GetCharacterFromLabel(int label) const {
    assert(label < size);
    return char_list[label];
  }

  int GetLabelFromCharacter(wchar_t c) {
    return char_to_label[c];
  }

  int GetSize() const {
    return size;
  }

private:
  int size;
  wchar_t *char_list;
  std::unordered_map<wchar_t, int> char_to_label;

  void ReadFromFile(std::istream& is) {
    std::string all_symbols_utf8;
    std::getline(is, all_symbols_utf8);
    std::wstring all_symbols;
    utf8::utf8to16(all_symbols_utf8.begin(),
                    all_symbols_utf8.end(),
                    std::back_inserter(all_symbols));
    size = all_symbols.size();
    char_list = new wchar_t[size];
    for (int i = 0; i < size; i++) {
      char_list[i] = all_symbols[i];
      char_to_label[char_list[i]] = i;
    }
  }

};

} // namespace ctc
} // namespace tensorflow

#endif //CTC_VOCABULARY_H
