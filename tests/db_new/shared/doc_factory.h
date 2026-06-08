// Copyright 2025-present the zvec project
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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>


namespace zvec::db_test {


struct DenseVectorCorpus {
  uint32_t dimension = 0;
  std::unordered_map<uint64_t, std::vector<float>> vectors;

  const std::vector<float> *Find(uint64_t doc_id) const {
    auto iter = vectors.find(doc_id);
    if (iter == vectors.end()) {
      return nullptr;
    }
    return &iter->second;
  }

  static Result<DenseVectorCorpus> LoadTsv(const std::string &path);
};


struct DenseVectorQuery {
  uint64_t query_id = 0;
  uint64_t target_doc_id = 0;
  std::vector<float> values;
};


struct DenseVectorQueryCorpus {
  uint32_t dimension = 0;
  std::vector<DenseVectorQuery> queries;

  const DenseVectorQuery *Find(uint64_t query_id) const {
    auto iter = std::find_if(
        queries.begin(), queries.end(),
        [query_id](const DenseVectorQuery &query) {
          return query.query_id == query_id;
        });
    if (iter == queries.end()) {
      return nullptr;
    }
    return &*iter;
  }

  static Result<DenseVectorQueryCorpus> LoadTsv(const std::string &path);
};


struct DenseVectorGroundTruth {
  uint32_t top_k = 0;
  std::unordered_map<uint64_t, uint64_t> target_doc_ids;
  std::unordered_map<uint64_t, std::vector<uint64_t>> neighbors;

  const std::vector<uint64_t> *FindNeighbors(uint64_t query_id) const {
    auto iter = neighbors.find(query_id);
    if (iter == neighbors.end()) {
      return nullptr;
    }
    return &iter->second;
  }

  static Result<DenseVectorGroundTruth> LoadTsv(const std::string &path);
};


struct DocFactoryOptions {
  std::string pk_prefix = "pk_";
  uint32_t array_length = 4;
  uint32_t sparse_dimension_if_zero = 4096;
  uint32_t sparse_non_zero_count = 8;
  bool use_null_pattern = false;
  uint64_t null_every = 5;
  const DenseVectorCorpus *dense_vector_corpus = nullptr;
};


namespace detail {


inline uint64_t Mix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}


inline uint64_t StableHash(const std::string &value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char ch : value) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
  return hash;
}


inline bool ShouldUseNull(const FieldSchema &field, uint64_t doc_id,
                          const DocFactoryOptions &options) {
  if (!options.use_null_pattern || !field.nullable() || field.is_vector_field()
      || options.null_every == 0) {
    return false;
  }
  return (doc_id + StableHash(field.name())) % options.null_every == 0;
}


inline bool StartsWithNumericToken(const std::string &line) {
  auto iter = std::find_if_not(line.begin(), line.end(), [](char ch) {
    return ch == ' ' || ch == '\t';
  });
  return iter != line.end() && *iter >= '0' && *iter <= '9';
}


inline float UnitFloat(uint64_t seed) {
  constexpr double kScale =
      1.0 / static_cast<double>(std::numeric_limits<uint32_t>::max());
  const auto mixed = Mix64(seed);
  return static_cast<float>(
      static_cast<double>(static_cast<uint32_t>(mixed)) * kScale * 2.0 - 1.0);
}


inline std::vector<float> MakeGeneratedDenseFp32Vector(uint64_t doc_id,
                                                       uint32_t dimension) {
  std::vector<float> values(dimension);
  double norm_squared = 0.0;

  for (uint32_t i = 0; i < dimension; ++i) {
    const auto seed = (doc_id << 32) ^ static_cast<uint64_t>(i);
    values[i] = UnitFloat(seed);
    norm_squared += static_cast<double>(values[i]) * values[i];
  }

  if (norm_squared == 0.0) {
    if (!values.empty()) {
      values[0] = 1.0f;
    }
    return values;
  }

  const auto inv_norm = static_cast<float>(1.0 / std::sqrt(norm_squared));
  for (auto &value : values) {
    value *= inv_norm;
  }
  return values;
}


inline std::vector<float> MakeDenseFp32Vector(
    const FieldSchema &field, uint64_t doc_id,
    const DocFactoryOptions &options) {
  if (options.dense_vector_corpus != nullptr) {
    const auto *fixture_vector = options.dense_vector_corpus->Find(doc_id);
    if (fixture_vector != nullptr &&
        fixture_vector->size() == field.dimension()) {
      return *fixture_vector;
    }
  }
  return MakeGeneratedDenseFp32Vector(doc_id, field.dimension());
}


inline std::vector<float16_t> ToFp16Vector(const std::vector<float> &values) {
  std::vector<float16_t> converted;
  converted.reserve(values.size());
  for (auto value : values) {
    converted.emplace_back(static_cast<float16_t>(value));
  }
  return converted;
}


inline std::vector<double> ToDoubleVector(const std::vector<float> &values) {
  std::vector<double> converted;
  converted.reserve(values.size());
  for (auto value : values) {
    converted.emplace_back(static_cast<double>(value));
  }
  return converted;
}


inline std::vector<int8_t> MakeDenseInt8Vector(uint64_t doc_id,
                                               uint32_t dimension) {
  std::vector<int8_t> values;
  values.reserve(dimension);
  for (uint32_t i = 0; i < dimension; ++i) {
    auto mixed = Mix64((doc_id << 32) ^ static_cast<uint64_t>(i));
    values.push_back(static_cast<int8_t>(static_cast<int>(mixed % 127) - 63));
  }
  return values;
}


inline std::vector<int16_t> MakeDenseInt16Vector(uint64_t doc_id,
                                                 uint32_t dimension) {
  std::vector<int16_t> values;
  values.reserve(dimension);
  for (uint32_t i = 0; i < dimension; ++i) {
    auto mixed = Mix64((doc_id << 32) ^ static_cast<uint64_t>(i));
    values.push_back(
        static_cast<int16_t>(static_cast<int>(mixed % 32767) - 16383));
  }
  return values;
}


inline std::vector<uint32_t> MakeDenseBinary32Vector(uint64_t doc_id,
                                                     uint32_t dimension) {
  std::vector<uint32_t> values;
  values.reserve(dimension);
  for (uint32_t i = 0; i < dimension; ++i) {
    values.push_back(static_cast<uint32_t>(
        Mix64((doc_id << 32) ^ static_cast<uint64_t>(i))));
  }
  return values;
}


inline std::vector<uint64_t> MakeDenseBinary64Vector(uint64_t doc_id,
                                                     uint32_t dimension) {
  std::vector<uint64_t> values;
  values.reserve(dimension);
  for (uint32_t i = 0; i < dimension; ++i) {
    values.push_back(Mix64((doc_id << 32) ^ static_cast<uint64_t>(i)));
  }
  return values;
}


inline uint32_t SparseDimension(const FieldSchema &field,
                                const DocFactoryOptions &options) {
  return field.dimension() == 0 ? options.sparse_dimension_if_zero
                                : field.dimension();
}


inline std::vector<uint32_t> MakeSparseIndices(
    const FieldSchema &field, uint64_t doc_id,
    const DocFactoryOptions &options) {
  const auto dimension = SparseDimension(field, options);
  const auto count = std::min(options.sparse_non_zero_count, dimension);
  std::vector<uint32_t> indices;
  indices.reserve(count);

  uint64_t salt = 0;
  while (indices.size() < count) {
    auto candidate = static_cast<uint32_t>(
        Mix64((doc_id << 32) ^ salt) % std::max<uint32_t>(dimension, 1));
    if (std::find(indices.begin(), indices.end(), candidate) ==
        indices.end()) {
      indices.push_back(candidate);
    }
    ++salt;
  }

  std::sort(indices.begin(), indices.end());
  return indices;
}


inline std::vector<float> MakeSparseFp32Values(size_t count, uint64_t doc_id) {
  std::vector<float> values;
  values.reserve(count);
  const auto scale = count == 0 ? 1.0f
                                : static_cast<float>(1.0 / std::sqrt(count));
  for (size_t i = 0; i < count; ++i) {
    const auto sign = (Mix64((doc_id << 32) ^ i) & 1ULL) == 0 ? 1.0f : -1.0f;
    values.push_back(sign * scale);
  }
  return values;
}


inline std::pair<std::vector<uint32_t>, std::vector<float>>
MakeSparseFp32Vector(const FieldSchema &field, uint64_t doc_id,
                     const DocFactoryOptions &options) {
  auto indices = MakeSparseIndices(field, doc_id, options);
  auto values = MakeSparseFp32Values(indices.size(), doc_id);
  return {std::move(indices), std::move(values)};
}


inline std::pair<std::vector<uint32_t>, std::vector<float16_t>>
MakeSparseFp16Vector(const FieldSchema &field, uint64_t doc_id,
                     const DocFactoryOptions &options) {
  auto fp32 = MakeSparseFp32Vector(field, doc_id, options);
  return {std::move(fp32.first), ToFp16Vector(fp32.second)};
}


inline uint32_t ArrayLength(uint64_t doc_id,
                            const DocFactoryOptions &options) {
  if (options.array_length == 0) {
    return 0;
  }
  return 1 + static_cast<uint32_t>(doc_id % options.array_length);
}


inline std::string MakeStringValue(const FieldSchema &field, uint64_t doc_id) {
  if (field.index_type() == IndexType::FTS || field.name() == "content") {
    return "doc_" + std::to_string(doc_id) + " group_" +
           std::to_string(doc_id % 10) + " bucket_" +
           std::to_string(doc_id % 4) +
           (doc_id % 2 == 0 ? " even alpha" : " odd beta");
  }
  if (field.name() == "name") {
    return "name_" + std::to_string(doc_id);
  }
  return field.name() + "_" + std::to_string(doc_id);
}


template <typename T>
inline std::vector<T> MakeNumericArray(uint64_t doc_id, uint32_t length) {
  std::vector<T> values;
  values.reserve(length);
  for (uint32_t i = 0; i < length; ++i) {
    values.push_back(static_cast<T>(doc_id + i));
  }
  return values;
}


inline void SetScalarOrArrayValue(Doc *doc, const FieldSchema &field,
                                  uint64_t doc_id,
                                  const DocFactoryOptions &options) {
  const auto length = ArrayLength(doc_id, options);

  switch (field.data_type()) {
    case DataType::BINARY:
      doc->set<std::string>(field.name(), "binary_" + std::to_string(doc_id));
      break;
    case DataType::STRING:
      doc->set<std::string>(field.name(), MakeStringValue(field, doc_id));
      break;
    case DataType::BOOL:
      doc->set<bool>(field.name(), doc_id % 2 == 0);
      break;
    case DataType::INT32:
      doc->set<int32_t>(field.name(), static_cast<int32_t>(doc_id));
      break;
    case DataType::INT64:
      doc->set<int64_t>(field.name(), static_cast<int64_t>(doc_id));
      break;
    case DataType::UINT32:
      doc->set<uint32_t>(field.name(), static_cast<uint32_t>(doc_id));
      break;
    case DataType::UINT64:
      doc->set<uint64_t>(field.name(), doc_id);
      break;
    case DataType::FLOAT:
      doc->set<float>(field.name(), static_cast<float>(doc_id) + 0.25f);
      break;
    case DataType::DOUBLE:
      doc->set<double>(field.name(), static_cast<double>(doc_id) + 0.125);
      break;
    case DataType::ARRAY_BINARY:
    case DataType::ARRAY_STRING: {
      std::vector<std::string> values;
      values.reserve(length);
      for (uint32_t i = 0; i < length; ++i) {
        values.push_back(field.name() + "_" + std::to_string(doc_id) + "_" +
                         std::to_string(i));
      }
      doc->set<std::vector<std::string>>(field.name(), std::move(values));
      break;
    }
    case DataType::ARRAY_BOOL: {
      std::vector<bool> values;
      values.reserve(length);
      for (uint32_t i = 0; i < length; ++i) {
        values.push_back((doc_id + i) % 2 == 0);
      }
      doc->set<std::vector<bool>>(field.name(), std::move(values));
      break;
    }
    case DataType::ARRAY_INT32:
      doc->set<std::vector<int32_t>>(
          field.name(), MakeNumericArray<int32_t>(doc_id, length));
      break;
    case DataType::ARRAY_INT64:
      doc->set<std::vector<int64_t>>(
          field.name(), MakeNumericArray<int64_t>(doc_id, length));
      break;
    case DataType::ARRAY_UINT32:
      doc->set<std::vector<uint32_t>>(
          field.name(), MakeNumericArray<uint32_t>(doc_id, length));
      break;
    case DataType::ARRAY_UINT64:
      doc->set<std::vector<uint64_t>>(
          field.name(), MakeNumericArray<uint64_t>(doc_id, length));
      break;
    case DataType::ARRAY_FLOAT:
      doc->set<std::vector<float>>(
          field.name(), MakeNumericArray<float>(doc_id, length));
      break;
    case DataType::ARRAY_DOUBLE:
      doc->set<std::vector<double>>(
          field.name(), MakeNumericArray<double>(doc_id, length));
      break;
    default:
      throw std::logic_error("unsupported scalar field [" + field.name() + "]");
  }
}


inline void SetVectorValue(Doc *doc, const FieldSchema &field, uint64_t doc_id,
                           const DocFactoryOptions &options) {
  switch (field.data_type()) {
    case DataType::VECTOR_BINARY32:
      doc->set<std::vector<uint32_t>>(
          field.name(), MakeDenseBinary32Vector(doc_id, field.dimension()));
      break;
    case DataType::VECTOR_BINARY64:
      doc->set<std::vector<uint64_t>>(
          field.name(), MakeDenseBinary64Vector(doc_id, field.dimension()));
      break;
    case DataType::VECTOR_FP16:
      doc->set<std::vector<float16_t>>(
          field.name(),
          ToFp16Vector(MakeDenseFp32Vector(field, doc_id, options)));
      break;
    case DataType::VECTOR_FP32:
      doc->set<std::vector<float>>(
          field.name(), MakeDenseFp32Vector(field, doc_id, options));
      break;
    case DataType::VECTOR_FP64:
      doc->set<std::vector<double>>(
          field.name(),
          ToDoubleVector(MakeDenseFp32Vector(field, doc_id, options)));
      break;
    case DataType::VECTOR_INT8:
      doc->set<std::vector<int8_t>>(
          field.name(), MakeDenseInt8Vector(doc_id, field.dimension()));
      break;
    case DataType::VECTOR_INT16:
      doc->set<std::vector<int16_t>>(
          field.name(), MakeDenseInt16Vector(doc_id, field.dimension()));
      break;
    case DataType::SPARSE_VECTOR_FP16:
      doc->set<std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
          field.name(), MakeSparseFp16Vector(field, doc_id, options));
      break;
    case DataType::SPARSE_VECTOR_FP32:
      doc->set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
          field.name(), MakeSparseFp32Vector(field, doc_id, options));
      break;
    default:
      throw std::logic_error("unsupported vector field [" + field.name() + "]");
  }
}


inline Result<std::vector<float>> ParseVectorValues(std::istringstream *stream,
                                                    uint32_t expected_dimension,
                                                    const std::string &path,
                                                    size_t line_number) {
  std::vector<float> values;
  float value = 0.0f;
  while (*stream >> value) {
    values.push_back(value);
  }

  if (values.empty()) {
    return tl::make_unexpected(Status::InvalidArgument(
        "empty vector in [", path, "] at line ", line_number));
  }
  if (expected_dimension != 0 && values.size() != expected_dimension) {
    return tl::make_unexpected(Status::InvalidArgument(
        "dimension mismatch in [", path, "] at line ", line_number));
  }
  return values;
}


}  // namespace detail


inline std::string MakePk(uint64_t doc_id,
                          const DocFactoryOptions &options = {}) {
  return options.pk_prefix + std::to_string(doc_id);
}


inline uint64_t ExtractDocId(const std::string &pk,
                             const DocFactoryOptions &options = {}) {
  if (pk.rfind(options.pk_prefix, 0) != 0) {
    throw std::invalid_argument("pk does not start with expected prefix");
  }
  return std::stoull(pk.substr(options.pk_prefix.size()));
}


inline Doc MakeDoc(const CollectionSchema &schema, uint64_t doc_id,
                   const DocFactoryOptions &options = {}) {
  Doc doc;
  doc.set_pk(MakePk(doc_id, options));

  for (const auto &field : schema.fields()) {
    if (detail::ShouldUseNull(*field, doc_id, options)) {
      doc.set_null(field->name());
      continue;
    }

    if (field->is_vector_field()) {
      detail::SetVectorValue(&doc, *field, doc_id, options);
    } else {
      detail::SetScalarOrArrayValue(&doc, *field, doc_id, options);
    }
  }
  return doc;
}


inline Doc MakeNullPatternDoc(const CollectionSchema &schema, uint64_t doc_id,
                              DocFactoryOptions options = {}) {
  options.use_null_pattern = true;
  return MakeDoc(schema, doc_id, options);
}


inline std::vector<Doc> MakeDocs(const CollectionSchema &schema,
                                 uint64_t start_doc_id, uint64_t count,
                                 const DocFactoryOptions &options = {}) {
  std::vector<Doc> docs;
  docs.reserve(count);
  for (uint64_t offset = 0; offset < count; ++offset) {
    docs.push_back(MakeDoc(schema, start_doc_id + offset, options));
  }
  return docs;
}


inline std::vector<float> MakeDenseQueryVector(
    const FieldSchema &field, uint64_t target_doc_id,
    const DocFactoryOptions &options = {}) {
  if (!field.is_dense_vector()) {
    throw std::invalid_argument("field is not a dense vector field");
  }
  return detail::MakeDenseFp32Vector(field, target_doc_id, options);
}


inline std::pair<std::vector<uint32_t>, std::vector<float>>
MakeSparseQueryVector(const FieldSchema &field, uint64_t target_doc_id,
                      const DocFactoryOptions &options = {}) {
  if (!field.is_sparse_vector()) {
    throw std::invalid_argument("field is not a sparse vector field");
  }
  return detail::MakeSparseFp32Vector(field, target_doc_id, options);
}


inline Status ValidateDocMatchesSchema(const CollectionSchema &schema,
                                       const Doc &doc,
                                       bool is_update = false) {
  auto schema_copy = std::make_shared<CollectionSchema>(schema);
  auto doc_copy = doc;
  return doc_copy.validate_and_sanitize(schema_copy, is_update);
}


inline Status ValidateDocMatchesPattern(
    const CollectionSchema &schema, const Doc &doc, uint64_t doc_id,
    const DocFactoryOptions &options = {}) {
  auto status = ValidateDocMatchesSchema(schema, doc);
  if (!status.ok()) {
    return status;
  }

  auto expected = MakeDoc(schema, doc_id, options);
  if (doc != expected) {
    return Status::InvalidArgument("doc[", doc.pk(),
                                   "] does not match generated pattern");
  }
  return Status::OK();
}


inline Result<DenseVectorCorpus> DenseVectorCorpus::LoadTsv(
    const std::string &path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    return tl::make_unexpected(
        Status::NotFound("failed to open dense vector corpus [", path, "]"));
  }

  DenseVectorCorpus corpus;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (!detail::StartsWithNumericToken(line)) {
      continue;
    }

    std::istringstream stream(line);
    uint64_t doc_id = 0;
    if (!(stream >> doc_id)) {
      return tl::make_unexpected(Status::InvalidArgument(
          "invalid doc id in [", path, "] at line ", line_number));
    }

    auto values = detail::ParseVectorValues(&stream, corpus.dimension, path,
                                            line_number);
    if (!values.has_value()) {
      return tl::make_unexpected(values.error());
    }
    if (corpus.dimension == 0) {
      corpus.dimension = static_cast<uint32_t>(values.value().size());
    }
    corpus.vectors[doc_id] = std::move(values.value());
  }

  if (corpus.vectors.empty()) {
    return tl::make_unexpected(
        Status::InvalidArgument("empty dense vector corpus [", path, "]"));
  }
  return corpus;
}


inline Result<DenseVectorQueryCorpus> DenseVectorQueryCorpus::LoadTsv(
    const std::string &path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    return tl::make_unexpected(
        Status::NotFound("failed to open dense vector queries [", path, "]"));
  }

  DenseVectorQueryCorpus corpus;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (!detail::StartsWithNumericToken(line)) {
      continue;
    }

    std::istringstream stream(line);
    DenseVectorQuery query;
    if (!(stream >> query.query_id >> query.target_doc_id)) {
      return tl::make_unexpected(Status::InvalidArgument(
          "invalid query row in [", path, "] at line ", line_number));
    }

    auto values = detail::ParseVectorValues(&stream, corpus.dimension, path,
                                            line_number);
    if (!values.has_value()) {
      return tl::make_unexpected(values.error());
    }
    if (corpus.dimension == 0) {
      corpus.dimension = static_cast<uint32_t>(values.value().size());
    }
    query.values = std::move(values.value());
    corpus.queries.push_back(std::move(query));
  }

  if (corpus.queries.empty()) {
    return tl::make_unexpected(
        Status::InvalidArgument("empty dense vector query corpus [", path,
                                "]"));
  }
  return corpus;
}


inline Result<DenseVectorGroundTruth> DenseVectorGroundTruth::LoadTsv(
    const std::string &path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    return tl::make_unexpected(Status::NotFound(
        "failed to open dense vector ground truth [", path, "]"));
  }

  DenseVectorGroundTruth ground_truth;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (!detail::StartsWithNumericToken(line)) {
      continue;
    }

    std::istringstream stream(line);
    uint64_t query_id = 0;
    uint64_t target_doc_id = 0;
    if (!(stream >> query_id >> target_doc_id)) {
      return tl::make_unexpected(Status::InvalidArgument(
          "invalid ground truth row in [", path, "] at line ", line_number));
    }

    std::vector<uint64_t> neighbors;
    uint64_t neighbor = 0;
    while (stream >> neighbor) {
      neighbors.push_back(neighbor);
    }
    if (neighbors.empty()) {
      return tl::make_unexpected(Status::InvalidArgument(
          "empty ground truth row in [", path, "] at line ", line_number));
    }

    if (ground_truth.top_k == 0) {
      ground_truth.top_k = static_cast<uint32_t>(neighbors.size());
    } else if (ground_truth.top_k != neighbors.size()) {
      return tl::make_unexpected(Status::InvalidArgument(
          "inconsistent ground truth top-k in [", path, "] at line ",
          line_number));
    }

    ground_truth.target_doc_ids[query_id] = target_doc_id;
    ground_truth.neighbors[query_id] = std::move(neighbors);
  }

  if (ground_truth.neighbors.empty()) {
    return tl::make_unexpected(Status::InvalidArgument(
        "empty dense vector ground truth [", path, "]"));
  }
  return ground_truth;
}


}  // namespace zvec::db_test
