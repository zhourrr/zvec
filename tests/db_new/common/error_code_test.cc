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


#include "db/common/error_code.h"
#include <gtest/gtest.h>


using namespace zvec;


TEST(ErrorCodeTest, Code) {
  EXPECT_EQ(PROXIMA_ZVEC_ERROR_CODE(Success).value(), 0);
  EXPECT_STREQ(PROXIMA_ZVEC_ERROR_CODE(Success).desc(), "Success");

  int code = PROXIMA_ZVEC_ERROR_CODE(RuntimeError);
  EXPECT_EQ(code, -1000);
}


TEST(ErrorCodeTest, What) {
  EXPECT_STREQ(ErrorCode::What(PROXIMA_ZVEC_ERROR_CODE(Success)), "Success");
  EXPECT_STREQ(ErrorCode::What(-999999), "");
}
