/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ctc/ctc_beam_entry.h"
#include "tensorflow/core/util/ctc/ctc_beam_scorer.h"
#include "tensorflow/core/util/ctc/ctc_vocabulary.h"
#include "lm/model.hh"

namespace {

using tensorflow::ctc::KenLMBeamScorer;
using tensorflow::ctc::ctc_beam_search::KenLMBeamState;
using tensorflow::ctc::Vocabulary;

const char test_sentence[] = "tomorrow it will rain";
// Input path for 'tomorrow it will rain'
const int test_labels_count = 108;
const int test_labels[] = {19,19,19,19,28,28,14,28,28,12,12,12,28,14,14,14,14,28,
                      28,17,17,28,28,28,17,17,17,17,28,14,14,14,28,28,28,28,
                      22,22,22,22,28,28,28,27,27,27,27,28,28,28,28,8,8,28,28,
                      28,19,19,19,28,28,28,27,28,22,22,22,28,28,28,8,28,28,28,
                      11,11,11,11,28,11,11,28,28,27,27,27,28,28,17,28,28,28,
                      28,0,0,28,28,28,8,8,28,28,28,13,13,13,13,28};
// Input path for 'tomorrow it will rain th'
const int test_labels_incomplete_count = 110;
const int test_labels_incomplete[] = {19,19,19,19,28,28,14,28,28,12,12,12,28,14,14,14,14,28,
                      28,17,17,28,28,28,17,17,17,17,28,14,14,14,28,28,28,28,
                      22,22,22,22,28,28,28,27,27,27,27,28,28,28,28,8,8,28,28,
                      28,19,19,19,28,28,28,27,28,22,22,22,28,28,28,8,28,28,28,
                      11,11,11,11,28,11,11,28,28,27,27,27,28,28,17,28,28,28,
                      28,0,0,28,28,28,8,8,28,28,28,13,13,13,13,28,19,7};
// Input path for 'tomorow it will rain'
const int test_labels_typo_count = 106;
const int test_labels_typo[] = {19,19,19,19,28,28,14,28,28,12,12,12,28,14,14,14,14,28,
                      28,28,28,28,17,17,17,17,28,14,14,14,28,28,28,28,
                      22,22,22,22,28,28,28,27,27,27,27,28,28,28,28,8,8,28,28,
                      28,19,19,19,28,28,28,27,28,22,22,22,28,28,28,8,28,28,28,
                      11,11,11,11,28,11,11,28,28,27,27,27,28,28,17,28,28,28,
                      28,0,0,28,28,28,8,8,28,28,28,13,13,13,13,28};

const char *kenlm_directory_path = "./tensorflow/core/util/ctc/testdata";
const char *vocabulary_path = "./tensorflow/core/util/ctc/testdata/vocabulary";
const char *model_path = "./tensorflow/core/util/ctc/testdata/kenlm-model.binary";

KenLMBeamScorer *createKenLMBeamScorer() {
  return new KenLMBeamScorer(kenlm_directory_path);
}

TEST(KenLMBeamSearch, Vocabulary) {

  const wchar_t char_list[] = L"abcdefghijklmnopqrstuvwxyz' ";
  Vocabulary vocabulary(char_list, 28);

  EXPECT_EQ(28, vocabulary.GetSize());
  EXPECT_EQ('b', vocabulary.GetCharacterFromLabel(1));
  EXPECT_EQ(4, vocabulary.GetLabelFromCharacter('e'));
  EXPECT_TRUE(vocabulary.IsBlankLabel(28));

  int previous_label = 0;
  int test_sentence_offset = 0;
  for (int i = 0; i < test_labels_count; i++) {
    int label = test_labels[i];
    if (label != previous_label && !vocabulary.IsBlankLabel(label)) {
      char returned_char = vocabulary.GetCharacterFromLabel(label);
      EXPECT_EQ(test_sentence[test_sentence_offset++], returned_char);
    }
    previous_label = label;
  }
}

TEST(KenLMBeamSearch, VocabularyFromFile) {

  Vocabulary vocabulary(vocabulary_path);

  EXPECT_EQ(28, vocabulary.GetSize());
  EXPECT_EQ('b', vocabulary.GetCharacterFromLabel(1));
  EXPECT_EQ(4, vocabulary.GetLabelFromCharacter('e'));
  EXPECT_TRUE(vocabulary.IsBlankLabel(28));
  EXPECT_TRUE(vocabulary.IsSpaceLabel(27));
}

TEST(KenLMBeamSearch, KenLMModel) {
  typedef lm::ngram::ProbingModel Model;

  lm::ngram::Config config;
  config.load_method = util::POPULATE_OR_READ;
  Model model(model_path, config);
  auto &vocabulary = model.GetVocabulary();

  Model::State states[2];
  states[0] = model.BeginSentenceState();

  float score = 0.0f;

  score += model.FullScore(states[0], vocabulary.Index("tomorrow"), states[1]).prob;
  score += model.FullScore(states[1], vocabulary.Index("it"), states[0]).prob;
  score += model.FullScore(states[0], vocabulary.Index("will"), states[1]).prob;
  score += model.FullScore(states[1], vocabulary.Index("rain"), states[0]).prob;
  score += model.FullScore(states[0], vocabulary.EndSentence(), states[1]).prob;

  EXPECT_NEAR(-4.21812, score, 0.0001);
}

std::string utf16to8(const std::wstring &word_utf16) {
    std::string encoded_word;
    utf8::utf16to8(word_utf16.begin(), word_utf16.end(), std::back_inserter(encoded_word));
    return encoded_word;
}

TEST(KenLMBeamSearch, KenLMModelWithUtf16) {
  typedef lm::ngram::ProbingModel Model;

  lm::ngram::Config config;
  config.load_method = util::POPULATE_OR_READ;
  Model model(model_path, config);
  auto &vocabulary = model.GetVocabulary();

  Model::State states[2];
  states[0] = model.BeginSentenceState();

  float score = 0.0f;

  score += model.FullScore(states[0], vocabulary.Index(utf16to8(L"tomorrow")), states[1]).prob;
  score += model.FullScore(states[1], vocabulary.Index(utf16to8(L"it")), states[0]).prob;
  score += model.FullScore(states[0], vocabulary.Index(utf16to8(L"will")), states[1]).prob;
  score += model.FullScore(states[1], vocabulary.Index(utf16to8(L"rain")), states[0]).prob;
  score += model.FullScore(states[0], vocabulary.EndSentence(), states[1]).prob;

  EXPECT_NEAR(-4.21812, score, 0.0001);
}

float ScoreBeam(KenLMBeamScorer *scorer, const int labels[], const int label_count) {
  KenLMBeamState states[2];
  scorer->InitializeState(&states[0]);

  int from_label = -1;
  float score = 0.0f;
  std::wstring incomplete_word;
  for (int i = 0; i < label_count; i++) {
    int to_label = labels[i];
    KenLMBeamState &from_state = states[i % 2];
    KenLMBeamState &to_state = states[(i + 1) % 2];
    
    scorer->ExpandState(from_state, from_label, &to_state, to_label);
    float new_score = scorer->GetStateExpansionScore(to_state, score);
    EXPECT_NEAR(new_score, to_state.score, 0.0001);
    if (incomplete_word == to_state.incomplete_word) {
      EXPECT_NEAR(score, new_score, 0.0001);
    }
    incomplete_word = to_state.incomplete_word;
    score = new_score;
    
    // Update from_label for next iteration
    from_label = to_label;
  }

  KenLMBeamState &endState = states[label_count % 2];
  scorer->ExpandStateEnd(&endState);
  score += scorer->GetStateEndExpansionScore(endState);
  EXPECT_NEAR(score, endState.score, 0.0001);
  EXPECT_NEAR(endState.language_model_score, endState.score, 0.0001);

  return score;
}

TEST(KenLMBeamSearch, PenalizeIncompleteWord) {
  KenLMBeamScorer *scorer = createKenLMBeamScorer();

  float log_prob_sound = ScoreBeam(scorer, test_labels, test_labels_count);
  float log_prob_incomplete = ScoreBeam(scorer, test_labels_incomplete, test_labels_incomplete_count);

  delete scorer;

  EXPECT_GT(log_prob_sound, log_prob_incomplete);
}

TEST(KenLMBeamSearch, PenalizeTypos) {
  KenLMBeamScorer *scorer = createKenLMBeamScorer();

  float log_prob_sound = ScoreBeam(scorer, test_labels, test_labels_count);
  float log_prob_typo = ScoreBeam(scorer, test_labels_typo, test_labels_typo_count);

  delete scorer;

  EXPECT_GT(log_prob_sound, log_prob_typo);
}

TEST(KenLMBeamSearch, ExpandState) {
  KenLMBeamScorer *scorer = createKenLMBeamScorer();

  float log_prob = ScoreBeam(scorer, test_labels, test_labels_count);

  delete scorer;

  EXPECT_NEAR(-4.21812, log_prob, 0.0001);
}

}  // namespace
