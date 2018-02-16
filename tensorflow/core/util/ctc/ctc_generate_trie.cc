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

#include "tensorflow/core/util/ctc/ctc_trie_node.h"
#include "tensorflow/core/util/ctc/ctc_vocabulary.h"
#include "lm/model.hh"
#include "utf8.h"

using namespace tensorflow::ctc;

typedef lm::ngram::ProbingModel Model;

lm::WordIndex GetWordIndex(const Model& model, const std::string& word) {
  lm::WordIndex vocab;
  vocab = model.GetVocabulary().Index(word);
  return vocab;
}

float ScoreWord(const Model& model, lm::WordIndex vocab) {
  Model::State in_state = model.NullContextState();
  Model::State out;
  lm::FullScoreReturn full_score_return;
  full_score_return = model.FullScore(in_state, vocab, out);
  return full_score_return.prob;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage " << argv[0]
              << " <kenlm_file_path>"
              << " <vocabulary_path>"
              << std::endl;
    return 1;
  }

  const char *kenlm_file_path = argv[1];
  const char *vocabulary_path = argv[2];

  lm::ngram::Config config;
  config.load_method = util::POPULATE_OR_READ;
  Model model(kenlm_file_path, config);

  Vocabulary vocabulary(vocabulary_path);

  TrieNode root(vocabulary.GetSize());

  std::string word;
  while (std::cin >> word) {
    lm::WordIndex vocab = GetWordIndex(model, word);
    float unigram_score = ScoreWord(model, vocab);
    std::wstring wide_word;
    utf8::utf8to16(word.begin(), word.end(), std::back_inserter(wide_word));
    root.Insert(wide_word.c_str(), [&vocabulary](wchar_t c) { 
                  return vocabulary.GetLabelFromCharacter(c);
                }, vocab, unigram_score);
  }

  root.WriteToStream(std::cout);
  return 0;
}
