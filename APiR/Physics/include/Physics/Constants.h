#pragma once
// Length   L         meters          m
// Volume   V                         m^3
// Density  D/rho                     kg / m^3
// Mass     M         kilograms       kg
// Time     T         seconds         s
// Force    F         newtons         kg * m / s^(-2)

// All constants must be written in SI
namespace Physics::Constants {
  constexpr double DEFAULT_TIME_STEP = 0.05;
  constexpr double GRAVITY_CONSTANT = 9.8;
}
