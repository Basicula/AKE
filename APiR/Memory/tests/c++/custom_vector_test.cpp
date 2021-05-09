#include "TestKernels.h"

#include <Memory/custom_vector.h>
#include <Memory/managed_ptr.h>
#include <Memory/device_ptr.h>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace {
  template<class T>
  void is_equal(const std::vector<T>& std_vec, const custom_vector<T>& vec) {
    ASSERT_EQ(std_vec.size(), vec.size());
    ASSERT_EQ(std_vec.capacity(), vec.capacity());
    for (size_t i = 0; i < std_vec.size(); ++i)
      EXPECT_EQ(std_vec[i], vec[i]);
    }
  }

TEST(custom_vector, default_constructor) {
  custom_vector<int> vec;
  std::vector<int> std_vec;
  is_equal(std_vec, vec);
  }

TEST(custom_vector, size_constructor) {
  const size_t size = 10;
  custom_vector<int> vec(size);
  std::vector<int> std_vec(size);
  is_equal(std_vec, vec);
  }

TEST(custom_vector, size_constructor_with_init_value) {
  const size_t size = 10;
  const int init_value = -123;
  custom_vector<int> vec(size, init_value);
  std::vector<int> std_vec(size, init_value);
  is_equal(std_vec, vec);
  }

TEST(custom_vector, foreach_is_supported) {
  const size_t size = 10;
  const int init_value = -123;
  custom_vector<int> vec(size, init_value);
  for (const auto val : vec)
    EXPECT_EQ(val, init_value);
  }

TEST(custom_vector, resize) {
  const size_t size1 = 10;
  const size_t size2 = 5;
  const size_t size3 = 15;
  const int init_value1 = -123;
  const int init_value2 = 123;
  custom_vector<int> vec;
  std::vector<int> std_vec;
  is_equal(std_vec, vec);

  vec.resize(size1);
  std_vec.resize(size1);
  is_equal(std_vec, vec);

  vec.clear();
  std_vec.clear();
  is_equal(std_vec, vec);

  vec.resize(size1, init_value1);
  std_vec.resize(size1, init_value1);
  is_equal(std_vec, vec);

  vec.resize(size2);
  std_vec.resize(size2);
  is_equal(std_vec, vec);

  vec.resize(size3, init_value2);
  std_vec.resize(size3, init_value2);
  is_equal(std_vec, vec);
  }

TEST(custom_vector, push_back) {
  const size_t size = 100;

  custom_vector<int> vec;
  std::vector<int> std_vec;

  for (int i = 0; i < size; ++i) {
    vec.push_back(i);
    std_vec.push_back(i);
    is_equal(std_vec, vec);
    }
  }

TEST(custom_vector, emplace_back) {
  const size_t size = 100;

  struct Point
    {
    double x, y, z;
    Point(double x, double y, double z)
      : x(x), y(y), z(z) {
      }
    bool operator==(const Point& i_other) const {
      return x == i_other.x && y == i_other.y && z == i_other.z;
      }
    };

  custom_vector<Point> vec;
  std::vector<Point> std_vec;

  for (int i = 0; i < size; ++i) {
    const auto vec_res = vec.emplace_back(i * 0.1, ( i + 1 ) * 0.1, ( i + 2 ) * 0.1);
    const auto std_vec_res = std_vec.emplace_back(i * 0.1, ( i + 1 ) * 0.1, ( i + 2 ) * 0.1);
    is_equal(std_vec, vec);
    EXPECT_EQ(vec_res, std_vec_res);
    }
  }

TEST(custom_vector, vector_of_vectors) {
  const size_t width = 100;
  const size_t height = 100;
  const int val = 11;
  custom_vector<custom_vector<int>> vec(width, custom_vector<int>(height, val));
  std::vector<std::vector<int>> std_vec(width, std::vector<int>(height, val));
  for (size_t i = 0; i < width; ++i)
    is_equal(std_vec[i], vec[i]);
  }

TEST(custom_vector, initializer_list) {
  std::vector<int> std_vec{ 1,2,3,4 };
  custom_vector<int> vec{ 1,2,3,4 };
  is_equal(std_vec, vec);
  }

TEST(custom_vector, managed_ptr_vector) {
  const size_t size = 10;
  managed_ptr<custom_vector<int>> managed_vector(size, 123);
  std::vector<int> vec(size, 123);
  is_equal(vec, *managed_vector.get());
  fill_vector(managed_vector.get(), size);
  std::iota(vec.begin(), vec.end(), 0);
  is_equal(vec, *managed_vector.get());
  }