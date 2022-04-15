#include "TestKernels.h"

#include "Macros.h"
#include "Memory/managed_ptr.h"

#include <gtest/gtest.h>

#include <vector>

TEST(managed_ptr, add_test) {
  managed_ptr<int> a(1), b(1);
  int c;
  add(a.get(), b.get(), c);
  EXPECT_EQ(2, c);
  }