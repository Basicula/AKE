#pragma once
#include "Geometry/BoundingBox.h"
#include "Rendering/Container.h"

#include <vector>

class KDTree : public Container
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

  public:
    KDTree() = default;

    HOSTDEVICE const Object* TraceRay(
      double& o_distance,
      const Ray& i_ray,
      const double i_far) const override;

    virtual void Update() override;

  private:
    void _Build(
      std::size_t i_start,
      std::size_t i_end);

    BoundingBox _BoundingBox(
      std::size_t i_start,
      std::size_t i_end);

  private:
    std::vector<KDNode> m_nodes;
  };