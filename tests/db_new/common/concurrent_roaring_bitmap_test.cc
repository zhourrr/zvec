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


#include "db/common/concurrent_roaring_bitmap.h"
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>


using namespace zvec;


namespace {


constexpr uint64_t kWideBase =
    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL;
constexpr uint32_t kWorkerNum = 4;


void AddRange(ConcurrentRoaringBitmap32 *bitmap, uint32_t start, uint32_t end) {
  std::vector<std::thread> threads;
  for (uint32_t worker = 0; worker < kWorkerNum; ++worker) {
    threads.emplace_back([worker, bitmap, start, end]() {
      if (start > end || worker > end - start) {
        return;
      }
      uint64_t value = start + worker;
      while (value <= end) {
        bitmap->add(static_cast<uint32_t>(value));
        value += kWorkerNum;
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
}


void AddRange(ConcurrentRoaringBitmap64 *bitmap, uint64_t start, uint64_t end) {
  std::vector<std::thread> threads;
  for (uint64_t worker = 0; worker < kWorkerNum; ++worker) {
    threads.emplace_back([worker, bitmap, start, end]() {
      if (start > end || worker > end - start) {
        return;
      }
      uint64_t value = start + worker;
      while (value <= end) {
        bitmap->add(value);
        if (end - value < kWorkerNum) {
          break;
        }
        value += kWorkerNum;
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
}


}  // namespace


TEST(ConcurrentRoaringBitmap32Test, Empty) {
  ConcurrentRoaringBitmap32 bitmap;

  EXPECT_EQ(bitmap.cardinality(), 0UL);
  EXPECT_EQ(bitmap.range_cardinality(0, 10), 0UL);
  EXPECT_FALSE(bitmap.contains(0));
}


TEST(ConcurrentRoaringBitmap64Test, Empty) {
  ConcurrentRoaringBitmap64 bitmap;

  EXPECT_EQ(bitmap.cardinality(), 0UL);
  EXPECT_EQ(bitmap.range_cardinality(kWideBase, kWideBase + 10), 0UL);
  EXPECT_FALSE(bitmap.contains(0));
  EXPECT_FALSE(bitmap.contains(kWideBase));
}


TEST(ConcurrentRoaringBitmap32Test, Clear) {
  ConcurrentRoaringBitmap32 bitmap;
  AddRange(&bitmap, 0, 1023);

  EXPECT_EQ(bitmap.cardinality(), 1024UL);
  EXPECT_TRUE(bitmap.contains(0));
  EXPECT_TRUE(bitmap.contains(1023));

  bitmap.clear();

  EXPECT_EQ(bitmap.cardinality(), 0UL);
  EXPECT_FALSE(bitmap.contains(0));
  EXPECT_FALSE(bitmap.contains(1023));
}


TEST(ConcurrentRoaringBitmap64Test, Clear) {
  ConcurrentRoaringBitmap64 bitmap;
  AddRange(&bitmap, kWideBase, kWideBase + 1023);

  EXPECT_EQ(bitmap.cardinality(), 1024UL);
  EXPECT_TRUE(bitmap.contains(kWideBase));
  EXPECT_TRUE(bitmap.contains(kWideBase + 1023));

  bitmap.clear();

  EXPECT_EQ(bitmap.cardinality(), 0UL);
  EXPECT_FALSE(bitmap.contains(kWideBase));
  EXPECT_FALSE(bitmap.contains(kWideBase + 1023));
}


TEST(ConcurrentRoaringBitmap32Test, RangeCardinality) {
  ConcurrentRoaringBitmap32 bitmap;
  AddRange(&bitmap, 0, 511);

  EXPECT_EQ(bitmap.cardinality(), 512UL);
  EXPECT_EQ(bitmap.range_cardinality(0, 511), 512UL);
  EXPECT_EQ(bitmap.range_cardinality(0, 127), 128UL);
  EXPECT_EQ(bitmap.range_cardinality(128, 255), 128UL);
  EXPECT_EQ(bitmap.range_cardinality(512, 1024), 0UL);
  EXPECT_TRUE(bitmap.contains(0));
  EXPECT_TRUE(bitmap.contains(511));
  EXPECT_FALSE(bitmap.contains(512));
}


TEST(ConcurrentRoaringBitmap64Test, RangeCardinality) {
  ConcurrentRoaringBitmap64 bitmap;
  AddRange(&bitmap, kWideBase, kWideBase + 511);

  EXPECT_EQ(bitmap.cardinality(), 512UL);
  EXPECT_EQ(bitmap.range_cardinality(kWideBase, kWideBase + 511), 512UL);
  EXPECT_EQ(bitmap.range_cardinality(kWideBase, kWideBase + 127), 128UL);
  EXPECT_EQ(bitmap.range_cardinality(kWideBase + 128, kWideBase + 255), 128UL);
  EXPECT_EQ(bitmap.range_cardinality(0, kWideBase - 1), 0UL);
  EXPECT_EQ(bitmap.range_cardinality(kWideBase + 512, kWideBase + 1024), 0UL);
  EXPECT_TRUE(bitmap.contains(kWideBase));
  EXPECT_TRUE(bitmap.contains(kWideBase + 511));
  EXPECT_FALSE(bitmap.contains(kWideBase + 512));
}


TEST(ConcurrentRoaringBitmap32Test, RemoveRange) {
  ConcurrentRoaringBitmap32 bitmap;
  AddRange(&bitmap, 0, 511);
  bitmap.remove_range_closed(128, 255);

  EXPECT_EQ(bitmap.cardinality(), 384UL);
  EXPECT_TRUE(bitmap.contains(127));
  EXPECT_FALSE(bitmap.contains(128));
  EXPECT_FALSE(bitmap.contains(255));
  EXPECT_TRUE(bitmap.contains(256));
  EXPECT_EQ(bitmap.range_cardinality(0, 511), 384UL);
  EXPECT_EQ(bitmap.range_cardinality(128, 255), 0UL);
}


TEST(ConcurrentRoaringBitmap64Test, RemoveRange) {
  ConcurrentRoaringBitmap64 bitmap;
  AddRange(&bitmap, kWideBase, kWideBase + 511);
  bitmap.remove_range_closed(kWideBase + 128, kWideBase + 255);

  EXPECT_EQ(bitmap.cardinality(), 384UL);
  EXPECT_TRUE(bitmap.contains(kWideBase + 127));
  EXPECT_FALSE(bitmap.contains(kWideBase + 128));
  EXPECT_FALSE(bitmap.contains(kWideBase + 255));
  EXPECT_TRUE(bitmap.contains(kWideBase + 256));
  EXPECT_EQ(bitmap.range_cardinality(kWideBase, kWideBase + 511), 384UL);
  EXPECT_EQ(bitmap.range_cardinality(kWideBase + 128, kWideBase + 255), 0UL);
}


TEST(ConcurrentRoaringBitmap64Test, UpgradeFrom32To64) {
  ConcurrentRoaringBitmap64 bitmap;
  AddRange(&bitmap, 0, 511);

  AddRange(&bitmap, kWideBase, kWideBase + 511);

  EXPECT_EQ(bitmap.cardinality(), 1024UL);
  EXPECT_EQ(bitmap.range_cardinality(0, 511), 512UL);
  EXPECT_EQ(bitmap.range_cardinality(kWideBase, kWideBase + 511), 512UL);
  EXPECT_TRUE(bitmap.contains(0));
  EXPECT_TRUE(bitmap.contains(511));
  EXPECT_TRUE(bitmap.contains(kWideBase));
  EXPECT_TRUE(bitmap.contains(kWideBase + 511));
  EXPECT_FALSE(bitmap.contains(512));
  EXPECT_FALSE(bitmap.contains(kWideBase + 512));
}


TEST(ConcurrentRoaringBitmap32Test, SerializeAndDeserialize) {
  ConcurrentRoaringBitmap32 bitmap;
  AddRange(&bitmap, 0, 511);

  std::string serialized;
  auto status = bitmap.serialize(&serialized);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(serialized.size(), bitmap.storage_size_in_bytes());

  ConcurrentRoaringBitmap32 restored;
  status = restored.deserialize(serialized);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(restored.cardinality(), 512UL);
  EXPECT_TRUE(restored.contains(0));
  EXPECT_TRUE(restored.contains(511));
  EXPECT_FALSE(restored.contains(512));
  EXPECT_EQ(restored.range_cardinality(0, 511), 512UL);
}


TEST(ConcurrentRoaringBitmap64Test, SerializeAndDeserialize) {
  const std::string path{"./concurrent_roaring_bitmap"};
  ailego::FileHelper::RemovePath(path.c_str());

  ConcurrentRoaringBitmap64 bitmap;
  AddRange(&bitmap, kWideBase, kWideBase + 511);

  ConcurrentRoaringBitmap64 restored;
  auto status = bitmap.serialize(path, false);
  ASSERT_TRUE(status.ok());

  status = restored.deserialize(path);
  ailego::FileHelper::RemovePath(path.c_str());
  ASSERT_TRUE(status.ok());

  EXPECT_EQ(restored.cardinality(), 512UL);
  EXPECT_TRUE(restored.contains(kWideBase));
  EXPECT_TRUE(restored.contains(kWideBase + 511));
  EXPECT_FALSE(restored.contains(kWideBase + 512));
  EXPECT_EQ(restored.range_cardinality(kWideBase, kWideBase + 511), 512UL);
}
