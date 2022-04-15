#include "Common/Randomizer.h"

Randomizer::Randomizer()
  : m_engine(std::random_device()())
{}

Randomizer::Randomizer(const unsigned int i_seed)
  : m_engine(i_seed)
{}

void Randomizer::SetSeed(const unsigned int i_seed)
{
  m_engine.seed(i_seed);
}
