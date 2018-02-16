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

// Collection of scoring classes that can be extended and provided to the
// CTCBeamSearchDecoder to incorporate additional scoring logic (such as a
// language model).
//
// To build a custom scorer extend and implement the pure virtual methods from
// BeamScorerInterface. The default CTC decoding behavior is implemented
// through BaseBeamScorer.

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_

#include "tensorflow/core/util/ctc/ctc_beam_entry.h"
#include "tensorflow/core/util/ctc/ctc_vocabulary.h"
#include "lm/model.hh"
#include "utf8.h"

#include <iostream>
#include <fstream>

namespace tensorflow {
namespace ctc {

using namespace ctc_beam_search;

// Base implementation of a beam scorer used by default by the decoder that can
// be subclassed and provided as an argument to CTCBeamSearchDecoder, if complex
// scoring is required. Its main purpose is to provide a thin layer for
// integrating language model scoring easily.
template <typename CTCBeamState>
class BaseBeamScorer {
 public:
  virtual ~BaseBeamScorer() {}
  // State initialization.
  virtual void InitializeState(CTCBeamState* root) const {}
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  virtual void ExpandState(const CTCBeamState& from_state, int from_label,
                           CTCBeamState* to_state, int to_label) const {}
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  virtual void ExpandStateEnd(CTCBeamState* state) const {}
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  virtual float GetStateExpansionScore(const CTCBeamState& state,
                                       float previous_score) const {
    return previous_score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  virtual float GetStateEndExpansionScore(const CTCBeamState& state) const {
    return 0;
  }
};

class KenLMBeamScorer : public BaseBeamScorer<KenLMBeamState> {
 public:
  typedef lm::ngram::ProbingModel Model;

  virtual ~KenLMBeamScorer() {
    delete model;
    delete trieRoot;
    delete vocabulary;
  }
  KenLMBeamScorer(const char *kenlm_directory_path)
                    : lm_weight(1.0f),
                      word_count_weight(0.0f) {

    std::string directory_path(kenlm_directory_path);
    const std::string model_path = directory_path + "/kenlm-model.binary";
    const std::string vocabulary_path = directory_path + "/vocabulary";
    const std::string trie_path = directory_path + "/trie";

    lm::ngram::Config config;
    config.load_method = util::POPULATE_OR_READ;
    model = new Model(model_path.c_str(), config);

    vocabulary = new Vocabulary(vocabulary_path.c_str());

    std::ifstream in;
    in.open(trie_path.c_str(), std::ios::in);
    TrieNode::ReadFromStream(in, trieRoot, vocabulary->GetSize());
    in.close();
  }

  // State initialization.
  void InitializeState(KenLMBeamState* root) const {
    root->language_model_score = 0.0f;
    root->score = 0.0f;
    root->delta_score = 0.0f;
    root->incomplete_word.clear();
    root->incomplete_word_trie_node = trieRoot;
    root->model_state = model->BeginSentenceState();
  }
  // ExpandState is called when expanding a beam to one of its children.
  // Called at most once per child beam. In the simplest case, no state
  // expansion is done.
  void ExpandState(const KenLMBeamState& from_state, int from_label,
                           KenLMBeamState* to_state, int to_label) const {
    CopyState(from_state, to_state);

    if (!vocabulary->IsSpaceLabel(to_label)) {
      to_state->incomplete_word += vocabulary->GetCharacterFromLabel(to_label);
      TrieNode *trie_node = from_state.incomplete_word_trie_node;

      // TODO replace with OOV unigram prob?
      // If we have no valid prefix we assume a very low log probability
      float min_unigram_score = -10.0f;
      // If prefix does exist
      if (trie_node != nullptr) {
        trie_node = trie_node->GetChildAt(to_label);
        to_state->incomplete_word_trie_node = trie_node;

        if (trie_node != nullptr) {
          min_unigram_score = trie_node->GetMinUnigramScore();
        }
      }
      // TODO try two options
      // 1) unigram score added up to language model scare
      // 2) langugage model score of (preceding_words + unigram_word)
      to_state->score = min_unigram_score + to_state->language_model_score;
      to_state->delta_score = to_state->score - from_state.score;

    } else {
      float lm_score_delta = ScoreIncompleteWord(from_state.model_state,
                            to_state->incomplete_word,
                            to_state->model_state);
      // Give fixed word bonus
      if (!IsOOV(to_state->incomplete_word)) {
        to_state->language_model_score += valid_word_count_weight;
      }
      to_state->language_model_score += word_count_weight;
      UpdateWithLMScore(to_state, lm_score_delta);
      ResetIncompleteWord(to_state);
    }
  }
  // ExpandStateEnd is called after decoding has finished. Its purpose is to
  // allow a final scoring of the beam in its current state, before resorting
  // and retrieving the TopN requested candidates. Called at most once per beam.
  void ExpandStateEnd(KenLMBeamState* state) const {
    float lm_score_delta = 0.0f;
    Model::State out;
    if (state->incomplete_word.size() > 0) {
      lm_score_delta += ScoreIncompleteWord(state->model_state,
                                            state->incomplete_word,
                                            out);
      ResetIncompleteWord(state);
      state->model_state = out;
    }
    lm_score_delta += model->FullScore(state->model_state,
                                      model->GetVocabulary().EndSentence(),
                                      out).prob;
    UpdateWithLMScore(state, lm_score_delta);
  }
  // GetStateExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandState. The score is
  // multiplied (log-addition) with the input score at the current step from
  // the network.
  //
  // The score returned should be a log-probability. In the simplest case, as
  // there's no state expansion logic, the expansion score is zero.
  float GetStateExpansionScore(const KenLMBeamState& state,
                                       float previous_score) const {
    return lm_weight * state.delta_score + previous_score;
  }
  // GetStateEndExpansionScore should be an inexpensive method to retrieve the
  // (cached) expansion score computed within ExpandStateEnd. The score is
  // multiplied (log-addition) with the final probability of the beam.
  //
  // The score returned should be a log-probability.
  float GetStateEndExpansionScore(const KenLMBeamState& state) const {
    return lm_weight * state.delta_score;
  }

  void SetLMWeight(float lm_weight) {
    this->lm_weight = lm_weight;
  }

  void SetWordCountWeight(float word_count_weight) {
    this->word_count_weight = word_count_weight; 
  }

  void SetValidWordCountWeight(float valid_word_count_weight) {
    this->valid_word_count_weight = valid_word_count_weight;
  }

 private:
  Vocabulary *vocabulary;
  TrieNode *trieRoot;
  Model *model;
  float lm_weight;
  float word_count_weight;
  float valid_word_count_weight;

  void UpdateWithLMScore(KenLMBeamState *state, float lm_score_delta) const {
    float previous_score = state->score;
    state->language_model_score += lm_score_delta;
    state->score = state->language_model_score;
    state->delta_score = state->language_model_score - previous_score;
  }

  void ResetIncompleteWord(KenLMBeamState *state) const {
    state->incomplete_word.clear();
    state->incomplete_word_trie_node = trieRoot;
  }

  bool IsOOV(const std::wstring& word) const {
    std::string encoded_word;
    utf8::utf16to8(word.begin(), word.end(), std::back_inserter(encoded_word));
    auto &vocabulary = model->GetVocabulary();
    return vocabulary.Index(encoded_word) == vocabulary.NotFound();
  }

  float ScoreIncompleteWord(const Model::State& model_state,
                            const std::wstring& word,
                            Model::State& out) const {
    lm::FullScoreReturn full_score_return;
    lm::WordIndex vocab;
    std::string encoded_word;
    utf8::utf16to8(word.begin(), word.end(), std::back_inserter(encoded_word));
    vocab = model->GetVocabulary().Index(encoded_word);
    full_score_return = model->FullScore(model_state, vocab, out);
    return full_score_return.prob;
  }

  void CopyState(const KenLMBeamState& from, KenLMBeamState* to) const {
    to->language_model_score = from.language_model_score;
    to->score = from.score;
    to->delta_score = from.delta_score;
    to->incomplete_word = from.incomplete_word;
    to->incomplete_word_trie_node = from.incomplete_word_trie_node;
    to->model_state = from.model_state;
  }

};

}  // namespace ctc
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CTC_CTC_BEAM_SCORER_H_
