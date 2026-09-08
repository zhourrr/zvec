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

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <zvec/db/schema.h>


namespace zvec::db_test {


constexpr uint32_t kDefaultDenseVectorDimension = 128;
constexpr uint32_t kDefaultSparseVectorDimension = 128;
constexpr uint32_t kDefaultQuantizedVectorDimension = 128;
constexpr uint32_t kMinRabitqVectorDimension = 64;
constexpr const char *kDenseVectorFieldName = "dense_fp32";
constexpr const char *kSparseVectorFieldName = "sparse_fp32";
constexpr const char *kQuantizedVectorFieldName = "quantized_dense_fp32";
constexpr const char *kFtsFieldName = "content";


inline IndexParams::Ptr MakeScalarIndexParams(
    bool enable_range_optimization = true,
    bool enable_extended_wildcard = false) {
  return std::make_shared<InvertIndexParams>(enable_range_optimization,
                                             enable_extended_wildcard);
}


inline IndexParams::Ptr MakeFlatIndexParams(
    MetricType metric_type = MetricType::IP,
    QuantizeType quantize_type = QuantizeType::UNDEFINED) {
  return std::make_shared<FlatIndexParams>(metric_type, quantize_type);
}


inline IndexParams::Ptr MakeHnswIndexParams(
    MetricType metric_type = MetricType::IP,
    int m = core_interface::kDefaultHnswNeighborCnt,
    int ef_construction = core_interface::kDefaultHnswEfConstruction,
    QuantizeType quantize_type = QuantizeType::UNDEFINED,
    bool use_contiguous_memory = false) {
  return std::make_shared<HnswIndexParams>(
      metric_type, m, ef_construction, quantize_type, use_contiguous_memory);
}


inline IndexParams::Ptr MakeHnswRabitqIndexParams(
    MetricType metric_type = MetricType::IP,
    int total_bits = core_interface::kDefaultRabitqTotalBits,
    int num_clusters = core_interface::kDefaultRabitqNumClusters,
    int m = core_interface::kDefaultHnswNeighborCnt,
    int ef_construction = core_interface::kDefaultHnswEfConstruction,
    int sample_count = 0) {
  return std::make_shared<HnswRabitqIndexParams>(
      metric_type, total_bits, num_clusters, m, ef_construction, sample_count);
}


inline IndexParams::Ptr MakeIVFIndexParams(
    MetricType metric_type = MetricType::IP, int n_list = 1024,
    int n_iters = 10, bool use_soar = false,
    QuantizeType quantize_type = QuantizeType::UNDEFINED) {
  return std::make_shared<IVFIndexParams>(metric_type, n_list, n_iters,
                                          use_soar, quantize_type);
}


inline IndexParams::Ptr MakeDiskAnnIndexParams(
    MetricType metric_type = MetricType::IP, int max_degree = 100,
    int list_size = 50, int pq_chunk_num = 0,
    QuantizeType quantize_type = QuantizeType::UNDEFINED) {
  return std::make_shared<DiskAnnIndexParams>(
      metric_type, max_degree, list_size, pq_chunk_num, quantize_type);
}


inline IndexParams::Ptr MakeVamanaIndexParams(
    MetricType metric_type = MetricType::IP,
    int max_degree = core_interface::kDefaultVamanaMaxDegree,
    int search_list_size = core_interface::kDefaultVamanaSearchListSize,
    float alpha = core_interface::kDefaultVamanaAlpha,
    bool saturate_graph = core_interface::kDefaultVamanaSaturateGraph,
    bool use_contiguous_memory = false, bool use_id_map = false,
    QuantizeType quantize_type = QuantizeType::UNDEFINED) {
  return std::make_shared<VamanaIndexParams>(
      metric_type, max_degree, search_list_size, alpha, saturate_graph,
      use_contiguous_memory, use_id_map, quantize_type);
}


inline IndexParams::Ptr MakeFtsIndexParams(
    std::string tokenizer_name = "standard",
    std::vector<std::string> filters = {"lowercase"},
    std::string extra_params = "") {
  return std::make_shared<FtsIndexParams>(
      std::move(tokenizer_name), std::move(filters), std::move(extra_params));
}


struct IndexParamCase {
  std::string name;
  IndexParams::Ptr index_params;
  uint32_t vector_dimension = kDefaultDenseVectorDimension;
};


inline std::vector<IndexParamCase> MakeScalarIndexParamCases() {
  return {
      {"range_off_wildcard_off", MakeScalarIndexParams(false, false)},
      {"range_on_wildcard_off", MakeScalarIndexParams(true, false)},
      {"range_off_wildcard_on", MakeScalarIndexParams(false, true)},
      {"range_on_wildcard_on", MakeScalarIndexParams(true, true)},
  };
}


inline std::vector<IndexParamCase> MakeDenseVectorIndexParamCases() {
  std::vector<IndexParamCase> cases = {
      {"flat_ip", MakeFlatIndexParams(MetricType::IP)},
      {"flat_l2", MakeFlatIndexParams(MetricType::L2)},
      {"flat_cosine", MakeFlatIndexParams(MetricType::COSINE)},
      {"flat_ip_fp16", MakeFlatIndexParams(MetricType::IP, QuantizeType::FP16)},
      {"flat_ip_int8", MakeFlatIndexParams(MetricType::IP, QuantizeType::INT8)},
      {"flat_ip_int4", MakeFlatIndexParams(MetricType::IP, QuantizeType::INT4)},
      {"hnsw_ip", MakeHnswIndexParams(MetricType::IP)},
      {"hnsw_l2", MakeHnswIndexParams(MetricType::L2)},
      {"hnsw_cosine", MakeHnswIndexParams(MetricType::COSINE)},
      {"hnsw_ip_fp16",
       MakeHnswIndexParams(MetricType::IP, 16, 100, QuantizeType::FP16)},
      {"hnsw_ip_int8",
       MakeHnswIndexParams(MetricType::IP, 16, 100, QuantizeType::INT8)},
      {"hnsw_ip_int4",
       MakeHnswIndexParams(MetricType::IP, 16, 100, QuantizeType::INT4)},
      {"ivf_ip", MakeIVFIndexParams(MetricType::IP, 1024, 10, false)},
      {"ivf_ip_soar", MakeIVFIndexParams(MetricType::IP, 1024, 10, true)},
      {"ivf_l2", MakeIVFIndexParams(MetricType::L2, 1024, 10, false)},
      {"ivf_cosine", MakeIVFIndexParams(MetricType::COSINE, 1024, 10, false)},
      {"ivf_ip_fp16",
       MakeIVFIndexParams(MetricType::IP, 1024, 10, false, QuantizeType::FP16)},
      {"ivf_ip_int8",
       MakeIVFIndexParams(MetricType::IP, 1024, 10, false, QuantizeType::INT8)},
      {"vamana_ip", MakeVamanaIndexParams(MetricType::IP)},
      {"vamana_ip_fp16",
       MakeVamanaIndexParams(MetricType::IP,
                             core_interface::kDefaultVamanaMaxDegree,
                             core_interface::kDefaultVamanaSearchListSize,
                             core_interface::kDefaultVamanaAlpha,
                             core_interface::kDefaultVamanaSaturateGraph, false,
                             false, QuantizeType::FP16)},
  };

#if DISKANN_SUPPORTED
  cases.push_back({"diskann_ip", MakeDiskAnnIndexParams(MetricType::IP)});
  cases.push_back(
      {"diskann_ip_fp16",
       MakeDiskAnnIndexParams(MetricType::IP, 100, 50, 0, QuantizeType::FP16)});
#endif

#if RABITQ_SUPPORTED
  cases.push_back({"hnsw_rabitq_ip", MakeHnswRabitqIndexParams(MetricType::IP),
                   kDefaultDenseVectorDimension});
  cases.push_back({"hnsw_rabitq_l2", MakeHnswRabitqIndexParams(MetricType::L2),
                   kDefaultDenseVectorDimension});
  cases.push_back({"hnsw_rabitq_cosine",
                   MakeHnswRabitqIndexParams(MetricType::COSINE),
                   kDefaultDenseVectorDimension});
#endif

  return cases;
}


inline std::vector<IndexParamCase> MakeDenseVectorRecallIndexParamCases() {
  std::vector<IndexParamCase> cases = {
      {"flat_l2", MakeFlatIndexParams(MetricType::L2)},
      {"hnsw_l2", MakeHnswIndexParams(MetricType::L2)},
      {"ivf_l2", MakeIVFIndexParams(MetricType::L2, 1024, 10, false)},
      {"vamana_l2", MakeVamanaIndexParams(MetricType::L2)},
  };

#if DISKANN_SUPPORTED
  cases.push_back({"diskann_l2", MakeDiskAnnIndexParams(MetricType::L2)});
#endif

#if RABITQ_SUPPORTED
  cases.push_back({"hnsw_rabitq_l2", MakeHnswRabitqIndexParams(MetricType::L2),
                   kDefaultDenseVectorDimension});
#endif

  return cases;
}


inline std::vector<IndexParamCase> MakeSparseVectorIndexParamCases() {
  return {
      {"flat_ip", MakeFlatIndexParams(MetricType::IP),
       kDefaultSparseVectorDimension},
      {"flat_ip_fp16", MakeFlatIndexParams(MetricType::IP, QuantizeType::FP16),
       kDefaultSparseVectorDimension},
      {"hnsw_ip", MakeHnswIndexParams(MetricType::IP),
       kDefaultSparseVectorDimension},
      {"hnsw_ip_fp16",
       MakeHnswIndexParams(MetricType::IP, 16, 100, QuantizeType::FP16),
       kDefaultSparseVectorDimension},
  };
}


inline std::vector<IndexParamCase> MakeQuantizedVectorIndexParamCases() {
  std::vector<IndexParamCase> cases = {
      {"flat_ip_fp16", MakeFlatIndexParams(MetricType::IP, QuantizeType::FP16)},
      {"flat_ip_int8", MakeFlatIndexParams(MetricType::IP, QuantizeType::INT8)},
      {"flat_ip_int4", MakeFlatIndexParams(MetricType::IP, QuantizeType::INT4)},
      {"hnsw_ip_fp16",
       MakeHnswIndexParams(MetricType::IP, 16, 100, QuantizeType::FP16)},
      {"hnsw_ip_int8",
       MakeHnswIndexParams(MetricType::IP, 16, 100, QuantizeType::INT8)},
      {"hnsw_ip_int4",
       MakeHnswIndexParams(MetricType::IP, 16, 100, QuantizeType::INT4)},
      {"ivf_ip_fp16",
       MakeIVFIndexParams(MetricType::IP, 1024, 10, false, QuantizeType::FP16)},
      {"ivf_ip_int8",
       MakeIVFIndexParams(MetricType::IP, 1024, 10, false, QuantizeType::INT8)},
  };

#if DISKANN_SUPPORTED
  cases.push_back(
      {"diskann_ip_fp16",
       MakeDiskAnnIndexParams(MetricType::IP, 100, 50, 0, QuantizeType::FP16)});
#endif

#if RABITQ_SUPPORTED
  cases.push_back({"hnsw_rabitq_ip", MakeHnswRabitqIndexParams(MetricType::IP),
                   kDefaultQuantizedVectorDimension});
#endif

  return cases;
}


inline std::vector<IndexParamCase> MakeFtsIndexParamCases() {
  return {
      {"standard_lowercase", MakeFtsIndexParams("standard")},
      {"whitespace_lowercase", MakeFtsIndexParams("whitespace")},
      {"whitespace_case_sensitive",
       MakeFtsIndexParams("whitespace", std::vector<std::string>{})},
  };
}


struct SchemaOptions {
  std::string name = "test_collection";
  bool nullable = false;
  bool enable_scalar_range_optimization = false;
  bool enable_scalar_extended_wildcard = false;
  uint64_t max_doc_count = MAX_DOC_COUNT_PER_SEGMENT;
  uint32_t dense_vector_dimension = kDefaultDenseVectorDimension;
  uint32_t sparse_vector_dimension = kDefaultSparseVectorDimension;
  uint32_t quantized_vector_dimension = kDefaultQuantizedVectorDimension;
  IndexParams::Ptr scalar_index_params = nullptr;
  IndexParams::Ptr dense_vector_index_params = nullptr;
  IndexParams::Ptr sparse_vector_index_params = nullptr;
  IndexParams::Ptr quantized_vector_index_params = nullptr;
  IndexParams::Ptr fts_index_params = nullptr;
};


namespace detail {


inline CollectionSchema::Ptr MakeEmptySchema(const SchemaOptions &options) {
  auto schema = std::make_shared<CollectionSchema>(options.name);
  schema->set_max_doc_count_per_segment(options.max_doc_count);
  return schema;
}


inline void AddField(const CollectionSchema::Ptr &schema,
                     const FieldSchema::Ptr &field) {
  auto status = schema->add_field(field);
  if (!status.ok()) {
    throw std::logic_error("failed to add test field [" + field->name() +
                           "]: " + status.message());
  }
}


inline IndexParams::Ptr ScalarIndexParamsFor(const SchemaOptions &options) {
  if (options.scalar_index_params) {
    return options.scalar_index_params;
  }
  return MakeScalarIndexParams(options.enable_scalar_range_optimization,
                               options.enable_scalar_extended_wildcard);
}


inline IndexParams::Ptr MaybeScalarIndexParams(const SchemaOptions &options,
                                               bool indexed) {
  if (!indexed) {
    return nullptr;
  }
  return ScalarIndexParamsFor(options);
}


inline IndexParams::Ptr DenseVectorIndexParamsFor(
    const SchemaOptions &options) {
  if (options.dense_vector_index_params) {
    return options.dense_vector_index_params;
  }
  return MakeFlatIndexParams();
}


inline IndexParams::Ptr SparseVectorIndexParamsFor(
    const SchemaOptions &options) {
  if (options.sparse_vector_index_params) {
    return options.sparse_vector_index_params;
  }
  return MakeFlatIndexParams();
}


inline IndexParams::Ptr QuantizedVectorIndexParamsFor(
    const SchemaOptions &options) {
  if (options.quantized_vector_index_params) {
    return options.quantized_vector_index_params;
  }
  return MakeHnswIndexParams(MetricType::IP, 16, 20, QuantizeType::FP16);
}


inline IndexParams::Ptr FtsIndexParamsFor(const SchemaOptions &options) {
  if (options.fts_index_params) {
    return options.fts_index_params;
  }
  return MakeFtsIndexParams("whitespace");
}


inline void AddScalarAndArrayFields(const CollectionSchema::Ptr &schema,
                                    const SchemaOptions &options,
                                    bool indexed) {
  auto index_params = MaybeScalarIndexParams(options, indexed);

  AddField(schema, std::make_shared<FieldSchema>(
                       "id", DataType::INT32, options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("name", DataType::STRING,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("age", DataType::UINT32,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("enabled", DataType::BOOL,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("count", DataType::INT64,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("rank", DataType::UINT64,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("score", DataType::FLOAT,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("rating", DataType::DOUBLE,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("binary", DataType::BINARY,
                                         options.nullable, index_params));

  AddField(schema,
           std::make_shared<FieldSchema>("array_binary", DataType::ARRAY_BINARY,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("array_string", DataType::ARRAY_STRING,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("array_bool", DataType::ARRAY_BOOL,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("array_int32", DataType::ARRAY_INT32,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("array_int64", DataType::ARRAY_INT64,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("array_uint32", DataType::ARRAY_UINT32,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("array_uint64", DataType::ARRAY_UINT64,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("array_float", DataType::ARRAY_FLOAT,
                                         options.nullable, index_params));
  AddField(schema,
           std::make_shared<FieldSchema>("array_double", DataType::ARRAY_DOUBLE,
                                         options.nullable, index_params));
}


inline void AddScalarAnchors(const CollectionSchema::Ptr &schema,
                             const SchemaOptions &options) {
  AddField(schema, std::make_shared<FieldSchema>("id", DataType::INT32,
                                                 options.nullable));
  AddField(schema, std::make_shared<FieldSchema>("name", DataType::STRING,
                                                 options.nullable));
}


inline void AddDenseVectorField(const CollectionSchema::Ptr &schema,
                                const SchemaOptions &options) {
  AddField(schema,
           std::make_shared<FieldSchema>(
               kDenseVectorFieldName, DataType::VECTOR_FP32,
               options.dense_vector_dimension, false,
               DenseVectorIndexParamsFor(options)));
}


inline void AddFlatDenseFp16VectorField(const CollectionSchema::Ptr &schema,
                                        const SchemaOptions &options) {
  AddField(schema,
           std::make_shared<FieldSchema>("dense_fp16", DataType::VECTOR_FP16,
                                         options.dense_vector_dimension, false,
                                         MakeFlatIndexParams(MetricType::IP)));
}


inline void AddFlatDenseInt8VectorField(const CollectionSchema::Ptr &schema,
                                        const SchemaOptions &options) {
  AddField(schema,
           std::make_shared<FieldSchema>("dense_int8", DataType::VECTOR_INT8,
                                         options.dense_vector_dimension, false,
                                         MakeFlatIndexParams(MetricType::IP)));
}


inline void AddSparseVectorField(const CollectionSchema::Ptr &schema,
                                 const SchemaOptions &options) {
  AddField(schema, std::make_shared<FieldSchema>(
                       kSparseVectorFieldName, DataType::SPARSE_VECTOR_FP32,
                       options.sparse_vector_dimension, false,
                       SparseVectorIndexParamsFor(options)));
}


inline void AddFlatSparseFp16VectorField(const CollectionSchema::Ptr &schema,
                                         const SchemaOptions &options) {
  AddField(schema, std::make_shared<FieldSchema>(
                       "sparse_fp16", DataType::SPARSE_VECTOR_FP16,
                       options.sparse_vector_dimension, false,
                       MakeFlatIndexParams(MetricType::IP)));
}


inline void AddQuantizedVectorField(const CollectionSchema::Ptr &schema,
                                    const SchemaOptions &options) {
  AddField(schema, std::make_shared<FieldSchema>(
                       kQuantizedVectorFieldName, DataType::VECTOR_FP32,
                       options.quantized_vector_dimension, false,
                       QuantizedVectorIndexParamsFor(options)));
}


inline void AddFtsField(const CollectionSchema::Ptr &schema,
                        const SchemaOptions &options) {
  AddField(schema,
           std::make_shared<FieldSchema>(kFtsFieldName, DataType::STRING, false,
                                         FtsIndexParamsFor(options)));
}


}  // namespace detail


inline CollectionSchema::Ptr MakeScalarSchema(SchemaOptions options = {}) {
  auto schema = detail::MakeEmptySchema(options);
  detail::AddScalarAndArrayFields(schema, options, false);
  return schema;
}


inline CollectionSchema::Ptr MakeScalarIndexSchema(SchemaOptions options = {}) {
  auto schema = detail::MakeEmptySchema(options);
  detail::AddScalarAndArrayFields(schema, options, true);
  return schema;
}


inline CollectionSchema::Ptr MakeDenseVectorSchema(SchemaOptions options = {}) {
  auto schema = detail::MakeEmptySchema(options);
  detail::AddScalarAnchors(schema, options);
  detail::AddDenseVectorField(schema, options);
  return schema;
}


inline CollectionSchema::Ptr MakeSparseVectorSchema(
    SchemaOptions options = {}) {
  auto schema = detail::MakeEmptySchema(options);
  detail::AddScalarAnchors(schema, options);
  detail::AddSparseVectorField(schema, options);
  return schema;
}


inline CollectionSchema::Ptr MakeQuantizedVectorSchema(
    SchemaOptions options = {}) {
  auto schema = detail::MakeEmptySchema(options);
  detail::AddScalarAnchors(schema, options);
  detail::AddQuantizedVectorField(schema, options);
  return schema;
}


inline CollectionSchema::Ptr MakeFtsSchema(SchemaOptions options = {}) {
  auto schema = detail::MakeEmptySchema(options);
  detail::AddScalarAnchors(schema, options);
  detail::AddFtsField(schema, options);
  return schema;
}


inline CollectionSchema::Ptr MakeAllTypesSchema(SchemaOptions options = {}) {
  auto schema = detail::MakeEmptySchema(options);
  detail::AddScalarAndArrayFields(schema, options, true);
  detail::AddDenseVectorField(schema, options);
  detail::AddFlatDenseFp16VectorField(schema, options);
  detail::AddFlatDenseInt8VectorField(schema, options);
  detail::AddSparseVectorField(schema, options);
  detail::AddFlatSparseFp16VectorField(schema, options);
  detail::AddQuantizedVectorField(schema, options);
  detail::AddFtsField(schema, options);
  return schema;
}


inline CollectionSchema::Ptr MakeDenseVectorSchemaForIndex(
    std::string name, const IndexParamCase &index_case,
    SchemaOptions options = {}) {
  options.name = std::move(name);
  options.dense_vector_dimension = index_case.vector_dimension;
  options.dense_vector_index_params = index_case.index_params;
  return MakeDenseVectorSchema(std::move(options));
}


inline CollectionSchema::Ptr MakeAllTypesSchemaForDenseVectorIndex(
    std::string name, const IndexParamCase &index_case,
    SchemaOptions options = {}) {
  options.name = std::move(name);
  options.dense_vector_dimension = index_case.vector_dimension;
  options.dense_vector_index_params = index_case.index_params;
  return MakeAllTypesSchema(std::move(options));
}


inline std::vector<CollectionSchema::Ptr> MakeDenseVectorSchemaVariants(
    const std::string &name_prefix, const IndexParamCase &index_case,
    SchemaOptions options = {}) {
  return {
      MakeDenseVectorSchemaForIndex(
          name_prefix + "_minimal_" + index_case.name, index_case, options),
      MakeAllTypesSchemaForDenseVectorIndex(
          name_prefix + "_all_types_" + index_case.name, index_case, options),
  };
}


inline std::vector<CollectionSchema::Ptr> MakeDenseVectorSchemas(
    const std::string &name_prefix,
    const std::vector<IndexParamCase> &index_cases,
    SchemaOptions options = {}) {
  std::vector<CollectionSchema::Ptr> schemas;
  schemas.reserve(index_cases.size() * 2);

  for (const auto &index_case : index_cases) {
    auto variants =
        MakeDenseVectorSchemaVariants(name_prefix, index_case, options);
    schemas.insert(schemas.end(), variants.begin(), variants.end());
  }

  return schemas;
}


inline std::vector<CollectionSchema::Ptr> MakeDenseVectorRecallSchemas(
    SchemaOptions options = {}) {
  return MakeDenseVectorSchemas("dense_recall",
                                MakeDenseVectorRecallIndexParamCases(),
                                std::move(options));
}


inline std::vector<CollectionSchema::Ptr> MakeDenseVectorIndexSchemas(
    SchemaOptions options = {}) {
  return MakeDenseVectorSchemas("dense_index",
                                MakeDenseVectorIndexParamCases(),
                                std::move(options));
}


}  // namespace zvec::db_test
