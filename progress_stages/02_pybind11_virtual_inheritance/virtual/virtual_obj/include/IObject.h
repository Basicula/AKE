#pragma once
#include <string>

class IObject
{
public:
    virtual std::string Get() const = 0;
    
    virtual ~IObject() = default;
};