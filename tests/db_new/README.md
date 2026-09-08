# DB Test Rewrite

`tests/db_new` is the new test suite for the db and index parts of the C++
engine. It is being built alongside the existing `tests/db` suite with the
intention of eventually replacing it.

During the rewrite, keep the old tests in place. New tests should be added here
with a simpler structure, clearer coverage boundaries, and focused assertions.
Once this suite covers the db and index behavior well enough, the old `tests/db`
suite can be removed and this directory can become the canonical db test suite.

Use `shared/` for anything reused across db tests, including CMake test helpers,
fixtures, matchers, and small C++ utilities.
