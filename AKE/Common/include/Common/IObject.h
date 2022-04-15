#pragma once
#include <string>

class IObject
{
public:
  virtual std::string Serialize() const = 0;
  virtual ~IObject() = default;
};