#pragma once
#include <IObject.h>

class Object : public IObject
{
public:
    Object(const std::string& i_str);
    
    virtual std::string Get() const override;
    
private:
    std::string m_str;
};