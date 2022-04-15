#pragma once
#include "Macros.h"
#include "Memory/custom_vector.h"

class A
  {
  public:
    HOSTDEVICE A(int a) : m_a(a) {
      }
    HOSTDEVICE void seta(int a) {
      m_a = a;
      }
    int geta()const {
      return m_a;
      }
    virtual int get() const = 0;
  protected:
    int m_a;
  };

class B : public A
  {
  public:
    HOSTDEVICE B(int b) : A(1), m_b(b) {
      }
    int getb() const {
      return m_b;
      }
    HOSTDEVICE void setb(int b) {
      m_b = b;
      }
    virtual int get() const {
      return m_a * m_b;
      }
  private:
    int m_b;
  };

void add(int* a, int* b, int& c);
void set(B* b);

void fill_vector(custom_vector<int>* iop_vector, size_t i_size);

