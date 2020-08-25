#pragma once
#include <Rendering/IRenderable.h>
#include <Geometry/BoundingBox.h>

#include <vector>
#include <memory>

using KDTreeObjects = std::vector<IRenderableSPtr>;

class KDNode
  {
  public:
    enum class Axis
      {
      X = 0,
      Y = 1,
      Z,
      };

    enum class NodeType
      {
      INTERNAL,
      LEAF,
      EMPTY
      };

  public:
    KDNode();
    KDNode(const KDTreeObjects& i_objects);

    void UpdateBBox();
    bool IntersectWithRay(
      IntersectionRecord& io_intersection, 
      const Ray& i_ray) const;

    void Clear();
    void Build(const KDTreeObjects& i_objects);

  private:
    void _Reset();

    static KDNode::Axis _DetectSplitingAxis(const BoundingBox& i_current_bbox);
    BoundingBox _GetNodeBoundingBox() const;

    bool _LeafIntersectWithRay(
      IntersectionRecord& io_intersection,
      const Ray& i_ray) const;

  private:
    NodeType m_type;
    BoundingBox m_bounding_box;
    KDTreeObjects m_objects;
    std::unique_ptr<KDNode> m_left;
    std::unique_ptr<KDNode> m_right;
  };