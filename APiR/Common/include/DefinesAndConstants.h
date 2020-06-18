#pragma once
#include <limits>

//-----------------------Math Common Constants-----------------------//

constexpr double Epsilon3D  = 1e-12;
constexpr double PI         = 3.14159265359;
constexpr double SQRT_2     = 1.41421356237;
constexpr double SQRT_3     = 1.73205080757;

// 1.79769e+308
constexpr double MAX_DOUBLE  = std::numeric_limits<double>::max(); 

// 2.22507e-308
constexpr double MIN_DOUBLE  = std::numeric_limits<double>::min();

// 2147483647
constexpr int MAX_INT        = std::numeric_limits<int>::max(); 

// -2147483648
constexpr int MIN_INT        = std::numeric_limits<int>::min(); 

//-----------------------Math Common Constants End-----------------------//

//-----------------------Physics Common Constants---------------------------//

// Length   L         meters          m
// Volume   V                         m^3
// Density  D/rho                     kg / m^3
// Mass     M         kilograms       kg
// Time     T         seconds         s
// Force    F         newtons         kg * m / s^(-2)

// All constants must be written in SI
constexpr double DEFAULT_TIME_STEP    = 0.05;
constexpr double GRAVITY_CONSTANT     = 9.8;

//-----------------------Physics Common Constants End-----------------------//