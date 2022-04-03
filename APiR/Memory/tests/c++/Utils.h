#pragma once
#include "Memory/custom_vector.h"

#include <gtest/gtest.h>

#include <vector>

template<class T>
void is_equal(const std::vector<T>& std_vec, const custom_vector<T>& vec) {
  ASSERT_EQ(std_vec.size(), vec.size());
  ASSERT_EQ(std_vec.capacity(), vec.capacity());
  for (size_t i = 0; i < std_vec.size(); ++i)
    EXPECT_EQ(std_vec[i], vec[i]);
  }