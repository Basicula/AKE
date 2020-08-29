#pragma once
#include <Rendering/IRenderable.h>
#include <Geometry/BoundingBox.h>

#include <vector>

class KDTree
  {
  private:
    enum class KDNodeType
      {
      INTERNAL = 0,
      LEAF = 1,
      EMPTY = 2,
      };

    struct KDNode
      {
      std::size_t start_obj_id;
      std::size_t end_obj_id;
      std::int64_t parent = -1;
      std::int64_t left;
      std::int64_t right;
      KDNodeType type;
      BoundingBox bounding_box;
      };

    using KDTreeObjects = std::vector<IRenderableSPtr>;
    using Nodes = std::vector<KDNode>;

  public:
    KDTree() = default;
    KDTree(KDTreeObjects&& i_objects);

    bool IntersectWithRay(
      IntersectionRecord& io_intersection,
      const Ray& i_ray) const;

    std::size_t KDTree::Size() const;

    void AddObject(IRenderableSPtr i_object);

    void Update();
    void Clear();

  private:
    void _Build(
      std::size_t i_start,
      std::size_t i_end);

    BoundingBox _BoundingBox(
      std::size_t i_start,
      std::size_t i_end);

  private:
    Nodes m_nodes;
    KDTreeObjects m_objects;
  };

inline std::size_t KDTree::Size() const
  {
  return m_objects.size();
  }