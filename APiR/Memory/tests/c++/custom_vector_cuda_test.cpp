#include "TestKernels.h"
#include "Utils.h"

#include <Memory/custom_vector.h>
#include <Memory/managed_ptr.h>

#include <gtest/gtest.h>

#include <numeric>

TEST(custom_vector, managed_ptr_vector) {
  const size_t size = 10;
  managed_ptr<custom_vector<int>> managed_vector(size, 123);
  std::vector<int> vec(size, 123);
  is_equal(vec, *managed_vector.get());
  fill_vector(managed_vector.get(), size);
  std::iota(vec.begin(), vec.end(), 0);
  is_equal(vec, *managed_vector.get());
  }