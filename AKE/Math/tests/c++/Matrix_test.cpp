#include <Math/SquareMatrix.h>
#include <gtest/gtest.h>

TEST(Matrix2x2, DefaultConstructor)
{
  Matrix2x2i m;
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_EQ(m(i, j), 0);
}

TEST(Matrix2x2, SpecializedConstructor)
{
  Matrix2x2i m(1, 2, 3, 4);
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(1, 0), 3);
  EXPECT_EQ(m(1, 1), 4);
}

TEST(Matrix2x2, Mutiply)
{
  Matrix2x2i m1(1, 2, 3, 4);
  Matrix2x2i m2(4, 3, 2, 1);
  const Matrix2x2i expected(8, 5, 20, 13);
  auto m3 = m1 * m2;
  EXPECT_EQ(m3, expected);
  m1 *= m2;
  EXPECT_EQ(m1, expected);
}

TEST(Matrix2x2, MutiplyOrthogonal)
{
  Matrix2x2i m1(0, -1, 1, 0);
  Matrix2x2i m2(0, 1, -1, 0);
  const Matrix2x2i expected(1, 0, 0, 1);
  const auto m3 = m1 * m2;
  EXPECT_EQ(m3, expected);
  const auto m4 = m2 * m1;
  EXPECT_EQ(m4, expected);
}

TEST(Matrix2x2, RotateVector90Degrees)
{
  Matrix2x2i rotation(0, -1, 1, 0);
  Vector2i vector(1, 0);
  const Vector2i expected(0, -1);
  rotation.ApplyLeft(vector);
  EXPECT_EQ(vector, expected);
}

TEST(Matrix2x2, RotateVector45Degrees)
{
  const double inv_sqrt_2 = 1.0 / sqrt(2);
  Matrix2x2d rotation(inv_sqrt_2, -inv_sqrt_2, inv_sqrt_2, inv_sqrt_2);
  Vector2d vector(1, 0);
  const Vector2d expected(inv_sqrt_2, -inv_sqrt_2);
  rotation.ApplyLeft(vector);
  EXPECT_EQ(vector, expected);
}

TEST(Matrix3x3, DefaultConstructor)
{
  Matrix3x3i m;
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_EQ(m(i, j), 0);
}

TEST(Matrix3x3, SpecializedConstructor)
{
  Matrix3x3i m(1, 2, 3, 4, 5, 6, 7, 8, 9);
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 2);
  EXPECT_EQ(m(0, 2), 3);
  EXPECT_EQ(m(1, 0), 4);
  EXPECT_EQ(m(1, 1), 5);
  EXPECT_EQ(m(1, 2), 6);
  EXPECT_EQ(m(2, 0), 7);
  EXPECT_EQ(m(2, 1), 8);
  EXPECT_EQ(m(2, 2), 9);
}

TEST(Matrix3x3, Mutiply)
{
  Matrix3x3i m1(1, 2, 3, 4, 5, 6, 7, 8, 9);
  Matrix3x3i m2(9, 8, 7, 6, 5, 4, 3, 2, 1);
  const Matrix3x3i expected(30, 24, 18, 84, 69, 54, 138, 114, 90);
  auto m3 = m1 * m2;
  EXPECT_EQ(m3, expected);
  m1 *= m2;
  EXPECT_EQ(m1, expected);
}

TEST(Matrix3x3, MutiplyOrthogonal)
{
  Matrix3x3i m1(0, -1, 0, 1, 0, 0, 0, 0, 1);
  Matrix3x3i m2(0, 1, 0, -1, 0, 0, 0, 0, 1);
  const Matrix3x3i expected(1, 0, 0, 0, 1, 0, 0, 0, 1);
  const auto m3 = m1 * m2;
  EXPECT_EQ(m3, expected);
  const auto m4 = m2 * m1;
  EXPECT_EQ(m4, expected);
}

TEST(Matrix3x3, RotateVector90Degrees)
{
  Matrix3x3i rotation(0, -1, 0, 1, 0, 0, 0, 0, 1);
  Vector3i vector(1, 0, 0);
  const Vector3i expected(0, -1, 0);
  rotation.ApplyLeft(vector);
  EXPECT_EQ(vector, expected);
}

TEST(Matrix3x3, RotateVector45Degrees)
{
  const double inv_sqrt_2 = 1.0 / sqrt(2);
  Matrix3x3d rotation(inv_sqrt_2, -inv_sqrt_2, 0.0, inv_sqrt_2, inv_sqrt_2, 0.0, 0.0, 0.0, 1.0);
  Vector3d vector(1.0, 0.0, 0.0);
  const Vector3d expected(inv_sqrt_2, -inv_sqrt_2, 0.0);
  rotation.ApplyLeft(vector);
  EXPECT_EQ(vector, expected);
}

TEST(SquareMatrix, DefaultConstructor)
{
  SquareMatrix<int, 5> m;
  for (size_t i = 0; i < 5; ++i)
    for (size_t j = 0; j < 5; ++j)
      EXPECT_EQ(m(i, j), 0);
}

TEST(SquareMatrix, Identity)
{
  SquareMatrix<int, 5> m;
  m.SetIdentity();
  for (size_t i = 0; i < 5; ++i)
    EXPECT_EQ(m(i, i), 1);
}

TEST(SquareMatrix, Transpose)
{
  Matrix3x3i m(1, 2, 3, 4, 5, 6, 7, 8, 9);
  const Matrix3x3i expected(1, 4, 7, 2, 5, 8, 3, 6, 9);
  m.Transpose();
  EXPECT_EQ(m, expected);
}