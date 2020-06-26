#include <KDNode.h>
#include <Intersection.h>
#include <ThreadPool.h>
#include <ParallelUtils.h>

KDNode::KDNode()
  {
  _Reset();
  }

KDNode::KDNode(const Objects& i_objects)
  {
  _Build(i_objects);
  }

void KDNode::UpdateBBox()
  {
  if (m_type == NodeType::EMPTY)
    return;

  if (m_type == NodeType::LEAF)
    m_bounding_box = _GetNodeBoundingBox();
  else
    {
    m_right->UpdateBBox();
    m_left->UpdateBBox();
    m_bounding_box = m_right->m_bounding_box;
    m_bounding_box.Merge(m_left->m_bounding_box);
    }
  }

bool KDNode::IntersectWithRay(
  IntersectionRecord& io_intersection,
  const Ray& i_ray) const
  {
  if (m_type == NodeType::EMPTY)
    return false;

  RayBoxIntersectionRecord ray_box_intersection;
  RayBoxIntersection(i_ray, m_bounding_box, ray_box_intersection);
  if ((ray_box_intersection.m_tmin < 0.0 && ray_box_intersection.m_tmax < 0.0) ||
      !ray_box_intersection.m_intersected ||
      ray_box_intersection.m_tmin > io_intersection.m_distance)
    return false;

  if (m_type == NodeType::LEAF)
    return _LeafIntersectWithRay(io_intersection, i_ray);

  const bool left = m_left->IntersectWithRay(io_intersection, i_ray);
  const bool right = m_right->IntersectWithRay(io_intersection, i_ray);
  return (left || right);
  }

KDNode::Axis KDNode::_DetectSplitingAxis(const BoundingBox& i_current_bbox)
  {
  const double x = i_current_bbox.DeltaX();
  const double y = i_current_bbox.DeltaY();
  const double z = i_current_bbox.DeltaZ();
  if (x > y && x > z)
    return KDNode::Axis::X;
  else if (y > z)
    return KDNode::Axis::Y;
  else
    return KDNode::Axis::Z;
  }

BoundingBox KDNode::_GetNodeBoundingBox() const
  {
  auto bounding_box = BoundingBox();
  for (const auto& object : m_objects)
    bounding_box.Merge(object->GetBoundingBox());
  return bounding_box;
  }

bool KDNode::_LeafIntersectWithRay(IntersectionRecord& io_intersection, const Ray& i_ray) const
  {
  if (m_type != NodeType::LEAF)
    return false;
  bool is_intersected = false;
  for (const auto& object : m_objects)
    is_intersected |= object->IntersectWithRay(io_intersection, i_ray);
  return is_intersected;
  }

void KDNode::_Build(const Objects& i_objects)
  {
  _Reset();
  if (i_objects.empty())
    return;

  m_objects = i_objects;
  m_bounding_box = _GetNodeBoundingBox();
  m_type = NodeType::LEAF;
  m_left.reset(new KDNode());
  m_right.reset(new KDNode());

  if (i_objects.size() == 1)
    return;

  auto axis = _DetectSplitingAxis(m_bounding_box);
  const auto& bbox_center = m_bounding_box.Center();
  Objects left, right;
  for (const auto& object : m_objects)
    if (object->GetBoundingBox().Center()[axis] < bbox_center[axis])
      left.push_back(object);
    else
      right.push_back(object);

  if (m_objects.size() == left.size()
    || m_objects.size() == right.size())
    return;

  m_type = NodeType::INTERNAL;
  m_left->_Build(left);
  m_right->_Build(right);
  }

void KDNode::_Reset()
  {
  m_type = NodeType::EMPTY;
  m_bounding_box = BoundingBox();
  m_objects.clear();
  m_left.release();
  m_right.release();
  }

void KDNode::Clear()
  {
  if (m_type == NodeType::EMPTY)
    return;
  if (m_left)
    m_left->Clear();
  if (m_right)
    m_right->Clear();
  _Reset();
  }