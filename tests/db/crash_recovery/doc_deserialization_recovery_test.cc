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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "db/index/storage/wal/wal_file.h"

namespace zvec {
namespace {

std::string ReadFileBytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(file), {});
}

std::string FindWal(const std::string &path) {
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (entry.path().extension() == ".wal") return entry.path().string();
  }
  return {};
}

std::map<std::string, std::string> ReadManifests(const std::string &path) {
  std::map<std::string, std::string> result;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (entry.path().filename().string().rfind("manifest", 0) == 0) {
      result.emplace(entry.path().string(),
                     ReadFileBytes(entry.path().string()));
    }
  }
  return result;
}

// Only called in ASSERT_EXIT children. Intentionally skip collection cleanup.
void WriteStringDocsAndExit(const std::string &path,
                            const std::vector<std::string> &ids,
                            const std::vector<std::string> &values,
                            bool malformed_record = false) {
  CollectionSchema schema("wal_recovery");
  if (!schema
           .add_field(
               std::make_shared<FieldSchema>("text", DataType::STRING, false))
           .ok()) {
    std::_Exit(1);
  }
  auto created = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  if (!created.has_value()) std::_Exit(2);
  auto collection = std::move(created).value();
  std::vector<Doc> docs;
  for (size_t i = 0; i < ids.size(); ++i) {
    Doc doc;
    doc.set_pk(ids[i]);
    doc.set<std::string>("text", values[i]);
    docs.push_back(std::move(doc));
  }
  auto result = collection->insert(docs);
  if (!result.has_value()) std::_Exit(3);
  for (const auto &status : result.value()) {
    if (!status.ok()) std::_Exit(4);
  }
  if (malformed_record) {
    auto wal = WalFile::Create(FindWal(path));
    if (wal->open(WalOptions{}) != 0 ||
        wal->append("invalid encoded document") != 0) {
      std::_Exit(5);
    }
  }
  std::_Exit(0);
}

class DocDeserializationDeathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
    ailego::FileHelper::RemovePath(path_.c_str());
  }
  void TearDown() override {
    ailego::FileHelper::RemovePath(path_.c_str());
  }
  const std::string path_{"doc_deserialization_recovery_test_db"};
};

TEST_F(DocDeserializationDeathTest, InvalidDocumentPayloadFailsRecovery) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(WriteStringDocsAndExit(path_, {"prefix"}, {"before"}, true),
              ::testing::ExitedWithCode(0), "");
  const auto wal_path = FindWal(path_);
  ASSERT_FALSE(wal_path.empty());
  const auto bytes = ReadFileBytes(wal_path);
  const auto manifests = ReadManifests(path_);
  auto opened = Collection::Open(path_, CollectionOptions{});
  ASSERT_FALSE(opened.has_value());
  EXPECT_NE(opened.error().message().find("Corrupt WAL document"),
            std::string::npos);
  EXPECT_EQ(ReadFileBytes(wal_path), bytes);
  EXPECT_EQ(ReadManifests(path_), manifests);
}

}  // namespace
}  // namespace zvec
