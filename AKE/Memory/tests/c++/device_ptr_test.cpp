#include "TestKernels.h"

#include "Macros.h"
#include "Memory/device_ptr.h"

#include <gtest/gtest.h>

#include <vector>

namespace {
  struct some_simple_struct
    {
    int value1 = -1;
    double value2 = 0.0;
    bool ok = false;

    HOSTDEVICE some_simple_struct(int v1, double v2, bool ok)
      : value1(v1)
      , value2(v2)
      , ok(ok) {
      }
    some_simple_struct() = default;
    some_simple_struct(const some_simple_struct&) = default;
    };
  }

TEST(device_ptr, simple_test) {
  device_ptr<int> d_a(1), d_b(1);
  int c;
  add(d_a.get(), d_b.get(), c);
  EXPECT_EQ(2, c);
  }

TEST(device_ptr, pointer_to_struct) {
  device_ptr<some_simple_struct> d_sss(12, 3.14, true);
  some_simple_struct sss;
  EXPECT_EQ(sss.value1, -1);
  EXPECT_EQ(sss.value2, 0.0);
  EXPECT_EQ(sss.ok, false);

  sss = d_sss.get_host_copy();
  EXPECT_EQ(sss.value1, 12);
  EXPECT_EQ(sss.value2, 3.14);
  EXPECT_EQ(sss.ok, true);
  }

TEST(device_ptr, inheritance) {
  device_ptr<B> d_b(10);
  set(d_b.get());
  auto b = d_b.get_host_copy();
  EXPECT_EQ(b.getb(), -2);
  EXPECT_EQ(b.geta(), 2);
  }