#pragma once
#include <limits>

template <typename Type>
Type Randomizer::Next()
{
  if constexpr (std::is_integral_v<Type>)
    return static_cast<Type>(m_engine() % std::numeric_limits<Type>::max());
  else
    return static_cast<Type>(m_engine());
}

template <typename Type>
Type Randomizer::Next(Type i_from, Type i_to)
{
  const auto unit_value = static_cast<double>(m_engine()) / m_engine.max();
  const auto range = i_to - i_from;
  const auto result = unit_value * range + i_from;
  return static_cast<Type>(result);
}

template <typename Type>
Type Randomizer::Get()
{
  std::random_device rd;
  std::mt19937_64 engine(rd());
  if constexpr (std::is_integral_v<Type>)
    return static_cast<Type>(engine() % std::numeric_limits<Type>::max());
  else
    return static_cast<Type>(engine());
}

template <typename Type>
Type Randomizer::Get(Type i_from, Type i_to)
{
  std::random_device rd;
  std::mt19937_64 engine(rd());
  const auto unit_value = static_cast<double>(engine()) / engine.max();
  const auto range = i_to - i_from;
  const auto result = unit_value * range + i_from;
  return static_cast<Type>(result);
}
