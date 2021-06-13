#include <Object.h>

Object::Object(const std::string& i_str)
    : m_str(i_str)
    {
    }
    
std::string Object::Get() const 
{
    return m_str;
}