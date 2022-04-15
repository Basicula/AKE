#pragma once
#include <random>

class Randomizer
{
public:
  Randomizer();
  explicit Randomizer(unsigned int i_seed);

  void SetSeed(unsigned int i_seed);

  template <typename Type>
  Type Next();

  template <typename Type>
  Type Next(Type i_from, Type i_to);

  template <typename Type>
  static Type Get();

  template <typename Type>
  static Type Get(Type i_from, Type i_to);

private:
  std::mt19937_64 m_engine;
};

#include "impl/RandomizerImpl.h"