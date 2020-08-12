#include <Visual/KDTree.h>

KDTree::KDTree()
  : m_objects()
  , m_root(new KDNode())
  {}

KDTree::KDTree(Objects&& i_objects)
  : m_objects(i_objects)
  , m_root(new KDNode(m_objects))
  {}