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


#include "db/index/segment/segment.h"
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/version_manager.h"
#include "shared/status_matchers.h"
#include "zvec/db/doc.h"
#include "zvec/db/options.h"
#include "zvec/db/schema.h"


namespace zvec {


namespace {


class SegmentTest : public testing::Test {
 protected:
  void SetUp() override {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
    zvec::ailego::MemoryLimitPool::get_instance().init(MIN_MEMORY_LIMIT_BYTES);

    collection_path_ =
        (std::filesystem::temp_directory_path() / UniqueTestDir()).string();
    FileHelper::RemoveDirectory(collection_path_);
    ASSERT_TRUE(FileHelper::CreateDirectory(collection_path_));

    const auto id_map_path =
        FileHelper::MakeFilePath(collection_path_, FileID::ID_FILE, 0);
    id_map_ =
        IDMap::CreateAndOpen(collection_name_, id_map_path, true, false);
    ASSERT_NE(id_map_, nullptr);

    delete_store_ = std::make_shared<DeleteStore>(collection_name_);

    schema_ = std::make_shared<CollectionSchema>(collection_name_);
    schema_->add_field(
        std::make_shared<FieldSchema>("id", DataType::INT32, false));
    schema_->add_field(
        std::make_shared<FieldSchema>("name", DataType::STRING, false));

    Version version;
    version.set_schema(*schema_);
    version.set_enable_mmap(false);

    auto version_manager = VersionManager::Create(collection_path_, version);
    ASSERT_TRUE(version_manager.has_value());
    version_manager_ = version_manager.value();

    options_.read_only_ = false;
    options_.enable_mmap_ = false;
    options_.max_buffer_size_ = DEFAULT_MAX_BUFFER_SIZE;
  }

  void TearDown() override {
    id_map_.reset();
    delete_store_.reset();
    version_manager_.reset();

    FileHelper::RemoveDirectory(collection_path_);
  }

  static std::string UniqueTestDir() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return "zvec_segment_test_" + std::to_string(now.count());
  }

  static Doc MakeDoc(uint64_t id) {
    Doc doc;
    doc.set_pk("pk_" + std::to_string(id));
    doc.set<int32_t>("id", static_cast<int32_t>(id));
    doc.set<std::string>("name", "name_" + std::to_string(id));
    return doc;
  }

  std::string collection_name_ = "test_segment";
  std::string collection_path_;
  IDMap::Ptr id_map_;
  DeleteStore::Ptr delete_store_;
  VersionManager::Ptr version_manager_;
  CollectionSchema::Ptr schema_;
  SegmentOptions options_;
};


TEST_F(SegmentTest, CreateInsertFetchAndScan) {
  auto result =
      Segment::CreateAndOpen(collection_path_, *schema_, 0, 0, id_map_,
                             delete_store_, version_manager_, options_);
  ASSERT_TRUE(result.has_value());
  auto segment = result.value();
  ASSERT_NE(segment, nullptr);
  EXPECT_EQ(segment->id(), 0);

  auto doc = MakeDoc(1);
  ZVEC_ASSERT_OK(segment->Insert(doc));
  EXPECT_EQ(doc.doc_id(), 0);
  EXPECT_EQ(segment->doc_count(), 1);

  auto fetched = segment->Fetch(0);
  ASSERT_NE(fetched, nullptr);
  EXPECT_EQ(fetched->doc_id(), 0);
  EXPECT_EQ(fetched->pk(), "pk_1");

  auto fetched_id = fetched->get<int32_t>("id");
  ASSERT_TRUE(fetched_id.has_value());
  EXPECT_EQ(fetched_id.value(), 1);

  auto fetched_name = fetched->get<std::string>("name");
  ASSERT_TRUE(fetched_name.has_value());
  EXPECT_EQ(fetched_name.value(), "name_1");

  auto batch = segment->fetch({"id", "name"}, 0);
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 2);

  auto id_scalar =
      std::dynamic_pointer_cast<arrow::Int32Scalar>(batch->values[0].scalar());
  ASSERT_NE(id_scalar, nullptr);
  EXPECT_EQ(id_scalar->value, 1);

  auto name_scalar =
      std::dynamic_pointer_cast<arrow::StringScalar>(batch->values[1].scalar());
  ASSERT_NE(name_scalar, nullptr);
  EXPECT_EQ(name_scalar->ToString(), "name_1");

  auto reader = segment->scan({"id", "name"});
  ASSERT_NE(reader, nullptr);

  std::shared_ptr<arrow::RecordBatch> scanned_batch;
  auto arrow_status = reader->ReadNext(&scanned_batch);
  ASSERT_TRUE(arrow_status.ok()) << arrow_status.ToString();
  ASSERT_NE(scanned_batch, nullptr);
  EXPECT_EQ(scanned_batch->num_rows(), 1);
  EXPECT_EQ(scanned_batch->num_columns(), 2);
  EXPECT_EQ(scanned_batch->column_name(0), "id");
  EXPECT_EQ(scanned_batch->column_name(1), "name");

  arrow_status = reader->ReadNext(&scanned_batch);
  ASSERT_TRUE(arrow_status.ok()) << arrow_status.ToString();
  EXPECT_EQ(scanned_batch, nullptr);
}


}  // namespace


}  // namespace zvec
