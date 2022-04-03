#include "Math/Matrix3.h"
#include "Math/Vector.h"

#include <gtest/gtest.h>

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
  Matrix3d matrix1{ 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 },
    matrix2{ 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
  const Matrix3d expected;
  const auto actual = matrix1 * matrix2;
  for (auto i = 0; i < 3; ++i)
    for (auto j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ(expected(i, j), actual(i, j));
}

TEST(Matrix3dMultiplicationSuite, Rotate90)
{
  Matrix3d matrix{ 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
  const Vector3d vector(1, 1, 0);
  const Vector3d expected(-1, 1, 0);
  const auto actual = matrix * vector;
  EXPECT_EQ(expected, actual);
}

TEST(Matrix3dMultiplicationSuite, Rotate45)
{
  const double sqrt2 = sqrt(2);
  Matrix3d matrix{ 1.0 / sqrt2, -1.0 / sqrt2, 0.0, 1.0 / sqrt2, 1.0 / sqrt2, 0.0, 0.0, 0.0, 1.0 };
  const Vector3d vector(1, 1, 0);
  const Vector3d expected(0, sqrt2, 0);
  const auto actual = matrix * vector;
  for (auto i = 0u; i < 3; ++i)
    EXPECT_DOUBLE_EQ(expected[i], actual[i]);
}

TEST(Matrix3dMultiplicationSuite, ApplyToVector)
{
  Matrix3d matrix{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

  Vector3d vec1(1.0, 2.0, 3.0);
  matrix.ApplyLeft(vec1);
  const Vector3d expected1(14.0, 32.0, 50.0);
  for (auto i = 0u; i < 3; ++i)
    EXPECT_DOUBLE_EQ(expected1[i], vec1[i]);

  Vector3d vec2(1.0, 2.0, 3.0);
  matrix.ApplyRight(vec2);
  const Vector3d expected2(30.0, 36.0, 42.0);
  for (auto i = 0u; i < 3; ++i)
    EXPECT_DOUBLE_EQ(expected2[i], vec2[i]);
}