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


#include "db/common/profiler.h"
#include <memory>
#include <gtest/gtest.h>


using namespace zvec;


TEST(ProfilerTest, Disabled) {
  Profiler profiler;

  EXPECT_FALSE(profiler.enabled());
  EXPECT_FALSE(profiler.enabled_debug());
  EXPECT_FALSE(profiler.enabled_trace());
  EXPECT_EQ(profiler.as_json_string(), "{}");

  profiler.start();
  EXPECT_EQ(profiler.open_stage("query"), 0);
  EXPECT_EQ(profiler.add("doc_count", 10), 0);
  EXPECT_EQ(profiler.close_stage(), 0);
  profiler.stop();

  EXPECT_EQ(profiler.as_json_string(), "{}");
}


TEST(ProfilerTest, TraceId) {
  Profiler profiler;

  profiler.set_trace_id("trace-1");

  EXPECT_TRUE(profiler.enabled());
  EXPECT_FALSE(profiler.enabled_debug());
  EXPECT_TRUE(profiler.enabled_trace());
  EXPECT_EQ(profiler.trace_id(), "trace-1");
}


TEST(ProfilerTest, RootValues) {
  Profiler profiler(true);

  profiler.start();
  EXPECT_EQ(profiler.add("doc_count", 10), 0);
  profiler.stop();

  const auto &root = profiler.root();
  ASSERT_TRUE(root.is_object());

  const auto &root_object = root.as_object();
  ASSERT_TRUE(root_object["doc_count"].is_integer());
  EXPECT_EQ(root_object["doc_count"].as_integer(), 10);
  ASSERT_TRUE(root_object["latency"].is_integer());
}


TEST(ProfilerTest, StageValues) {
  Profiler profiler(true);

  profiler.start();
  EXPECT_EQ(profiler.open_stage("search"), 0);
  EXPECT_EQ(profiler.add("matched_count", 3), 0);
  EXPECT_EQ(profiler.close_stage(), 0);
  profiler.stop();

  const auto &root_object = profiler.root().as_object();
  const auto stage = root_object["search"];
  ASSERT_TRUE(stage.is_object());

  const auto &stage_object = stage.as_object();
  ASSERT_TRUE(stage_object["matched_count"].is_integer());
  EXPECT_EQ(stage_object["matched_count"].as_integer(), 3);
  ASSERT_TRUE(stage_object["latency"].is_integer());
  ASSERT_TRUE(root_object["latency"].is_integer());
}


TEST(ProfilerTest, Errors) {
  Profiler profiler(true);

  EXPECT_EQ(profiler.add("doc_count", 10),
            PROXIMA_ZVEC_ERROR_CODE(RuntimeError));
  EXPECT_EQ(profiler.open_stage("search"),
            PROXIMA_ZVEC_ERROR_CODE(RuntimeError));
  EXPECT_EQ(profiler.close_stage(), PROXIMA_ZVEC_ERROR_CODE(RuntimeError));

  profiler.start();

  EXPECT_EQ(profiler.open_stage(""), PROXIMA_ZVEC_ERROR_CODE(RuntimeError));
}


TEST(ProfilerTest, ScopedProfilerStage) {
  auto profiler = std::make_shared<Profiler>(true);

  profiler->start();
  {
    ScopedProfilerStage stage(profiler, "filter");

    EXPECT_EQ(profiler->add("passed_count", 7), 0);
  }
  profiler->stop();

  const auto &root_object = profiler->root().as_object();
  const auto stage_value = root_object["filter"];
  ASSERT_TRUE(stage_value.is_object());

  const auto &stage_object = stage_value.as_object();
  ASSERT_TRUE(stage_object["passed_count"].is_integer());
  EXPECT_EQ(stage_object["passed_count"].as_integer(), 7);
  ASSERT_TRUE(stage_object["latency"].is_integer());
}


TEST(ProfilerTest, ScopedLatency) {
  auto profiler = std::make_shared<Profiler>(true);

  profiler->start();
  {
    ScopedLatency latency("elapsed", profiler);
  }
  profiler->stop();

  const auto &root_object = profiler->root().as_object();
  ASSERT_TRUE(root_object["elapsed"].is_integer());
  ASSERT_TRUE(root_object["latency"].is_integer());
}
