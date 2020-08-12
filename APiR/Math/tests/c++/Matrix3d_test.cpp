#include <gtest/gtest.h>

#include <Math/Matrix3.h>
#include <Math/Vector.h>

TEST(Matrix3dMultiplicationSuite, IdentityAndVector)
  {
  Matrix3d matrix;
  Vector3d vector(1, 2, 3);
  EXPECT_EQ(vector, matrix * vector);
  }

TEST(Matrix3dMultiplicationSuite, MatrixAndMatrix)
  {
  Matrix3d matrix1, matrix2;
  const Matrix3d expected;
  const auto actual = matrix1 * matrix2;
  for (auto i = 0; i < 3; ++i)
    for (auto j = 0; j < 3; ++j)
    EXPECT_DOUBLE_EQ(expected(i, j), actual(i, j));
  }

TEST(Matrix3dMultiplicationSuite, CheckRotationOrthogonality)
  {
  Matrix3d matrix1
    {
    0.0, -1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 0.0, 1.0
    }, 
    matrix2
    {
    0.0, 1.0, 0.0,
    -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0
    };
  const Matrix3d expected;
  const auto actual = matrix1 * matrix2;
  for (auto i = 0; i < 3; ++i)
    for (auto j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ(expected(i, j), actual(i, j));
  }

TEST(Matrix3dMultiplicationSuite, Rotate90)
  {
  Matrix3d matrix
    {
    0.0, -1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 0.0, 1.0
    };
  const Vector3d vector(1, 1, 0);
  const Vector3d expected(-1, 1, 0);
  const auto actual = matrix * vector;
  EXPECT_EQ(expected, actual);
  }

TEST(Matrix3dMultiplicationSuite, Rotate45)
  {
  const double sqrt2 = sqrt(2);
  Matrix3d matrix
    {
    1.0 / sqrt2, -1.0 / sqrt2, 0.0,
    1.0 / sqrt2, 1.0 / sqrt2, 0.0,
    0.0, 0.0, 1.0
    };
  const Vector3d vector(1, 1, 0);
  const Vector3d expected(0, sqrt2, 0);
  const auto actual = matrix * vector;
  for (auto i = 0u; i < 3; ++i)
    EXPECT_DOUBLE_EQ(expected[i], actual[i]);
  }