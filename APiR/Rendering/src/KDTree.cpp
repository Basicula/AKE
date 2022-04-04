#include "Rendering/KDTree.h"

#include "Geometry.3D/Intersection.h"

namespace {
  double Volume(const BoundingBox& i_bbox) { return i_bbox.DeltaX() * i_bbox.DeltaY() * i_bbox.DeltaZ(); }

  struct SAHSplittingInfo
  {
    std::size_t axis;
    std::size_t left_count;
    std::size_t right_count;
    double split;
    double sah;
  };

  bool comparator_bbox_min(const Object* i_first, const Object* i_second)
  {
    const auto& bbox_first = i_first->GetBoundingBox();
    const auto& bbox_second = i_second->GetBoundingBox();
    return bbox_first.GetMin() < bbox_second.GetMin();
  };

  bool comparator_bbox_max(const Object* i_first, const Object* i_second)
  {
    const auto& bbox_first = i_first->GetBoundingBox();
    const auto& bbox_second = i_second->GetBoundingBox();
    return bbox_first.GetMax() < bbox_second.GetMax();
  };

  void UpdateSAH(SAHSplittingInfo& o_splitting_info,
                 const BoundingBox& i_bbox,
                 double i_split,
                 std::size_t i_axis,
                 std::size_t i_left_cnt,
                 std::size_t i_right_cnt)
  {
    const double volume = Volume(i_bbox);
    const double left_to_right_ratio = (i_split - i_bbox.GetMin()[i_axis]) / (i_bbox.GetMax()[i_axis] - i_split);
    const double volume_right = volume / (1 + left_to_right_ratio);
    const double volume_left = volume - volume_right;
    const double sah = i_left_cnt * volume_left + i_right_cnt * volume_right;
    if (sah < o_splitting_info.sah) {
      o_splitting_info.sah = sah;
      o_splitting_info.split = i_split;
      o_splitting_info.axis = i_axis;
      o_splitting_info.left_count = i_left_cnt;
      o_splitting_info.right_count = i_right_cnt;
    }
  };
}

void KDTree::_Build(std::size_t i_start, std::size_t i_end)
{
  m_nodes.emplace_back();
  auto& curr_node = m_nodes.back();

  const auto curr_bbox = _BoundingBox(i_start, i_end);
  curr_node.bounding_box = curr_bbox;
  curr_node.start_obj_id = i_start;
  curr_node.end_obj_id = i_end;

  if (i_end == i_start + 1) {
    curr_node.type = KDNodeType::LEAF;
    return;
  }

  SAHSplittingInfo splitting_info{ 0, 0, 0, INFINITY, INFINITY };
  for (auto axis = 0u; axis < 3; ++axis) {
    std::sort(m_objects.begin() + i_start, m_objects.begin() + i_end, comparator_bbox_min);
    for (auto i = i_start + 1; i < i_end; ++i) {
      const std::size_t left_side_cnt = i - i_start;
      const std::size_t right_side_cnt = i_end - i;
      double split = m_objects[i]->GetBoundingBox().GetMin()[axis];
      if (split == curr_bbox.GetMin()[axis])
        continue;

      UpdateSAH(splitting_info, curr_bbox, split, axis, left_side_cnt, right_side_cnt);
    }

    std::sort(m_objects.begin() + i_start, m_objects.begin() + i_end, comparator_bbox_max);
    for (long long i = i_end - 2; i >= static_cast<long long>(i_start); --i) {
      const std::size_t left_side_cnt = i + 1 - i_start;
      const std::size_t right_side_cnt = i_end - i - 1;
      double split = m_objects[i]->GetBoundingBox().GetMax()[axis];
      if (split == curr_bbox.GetMax()[axis])
        continue;

      UpdateSAH(splitting_info, curr_bbox, split, axis, left_side_cnt, right_side_cnt);
    }
  }

  const auto left = static_cast<std::int64_t>(m_nodes.size());
  _Build(i_start, i_start + splitting_info.left_count);
  curr_node.left = left;
  m_nodes[left].parent = left - 1;

  const auto right = static_cast<std::int64_t>(m_nodes.size());
  _Build(i_start + splitting_info.left_count, i_end);
  curr_node.right = right;
  m_nodes[right].parent = left - 1;
}

BoundingBox KDTree::_BoundingBox(std::size_t i_start, std::size_t i_end)
{
  auto res = BoundingBox();
  for (auto i = i_start; i < i_end; ++i)
    res.Merge(m_objects[i]->GetBoundingBox());
  return res;
}

const Object* KDTree::TraceRay(double& o_distance, const Ray& i_ray, const double i_far) const
{
  const Object* intersected_object = nullptr;
  long long prev_id = -1, curr_id = 0;
  auto far = i_far;
  while (curr_id != -1) {
    auto& node = m_nodes[curr_id];
    // came from child
    if (curr_id < prev_id) {
      if (prev_id == node.left) {
        prev_id = curr_id;
        curr_id = node.right;
      } else if (prev_id == node.right) {
        prev_id = curr_id;
        curr_id = node.parent;
      }
      continue;
    }

    RayBoxIntersectionRecord ray_box_intersection;
    RayBoxIntersection(i_ray, node.bounding_box, ray_box_intersection);
    if ((ray_box_intersection.m_tmin < 0.0 && ray_box_intersection.m_tmax < 0.0) ||
        !ray_box_intersection.m_intersected || ray_box_intersection.m_tmin > far) {
      prev_id = curr_id;
      curr_id = node.parent;
      continue;
    }

    if (node.type == KDNodeType::LEAF) {
      for (auto i = node.start_obj_id; i < node.end_obj_id; ++i)
        if (m_objects[i]->IntersectWithRay(o_distance, i_ray, far)) {
          if (o_distance < far) {
            intersected_object = m_objects[i];
            far = o_distance;
          }
        }
      prev_id = curr_id;
      curr_id = node.parent;
      continue;
    }
    prev_id = curr_id;
    curr_id = node.left;
  }
  return intersected_object;
}

void KDTree::Update()
{
  m_nodes.clear();
  m_nodes.reserve(static_cast<size_t>(1 + m_objects.size() * log2(static_cast<double>(m_objects.size()))));
  _Build(0, m_objects.size());
}