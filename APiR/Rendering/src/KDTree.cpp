#include <Rendering/KDTree.h>

KDTree::KDTree()
  : m_objects()
  , m_root()
  {}

KDTree::KDTree(KDTreeObjects&& i_objects)
  : m_objects(i_objects)
  , m_root(m_objects)
  {}