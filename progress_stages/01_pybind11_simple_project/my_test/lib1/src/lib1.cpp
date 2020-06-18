#include <lib1.h>

Adder::Adder()
  : m_res(0)
  {
  }
  
int Adder::get()
{
    return m_res;
}

void Adder::push(int i_num)
{
    m_res+=i_num;
}