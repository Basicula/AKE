#pragma once

#include <Sphere.h>
#include <Ray.h>

#include <algorithm>

bool IntersectRayWithSphere(Vector3d& o_intersection, const Ray& i_ray, const Sphere& i_sphere);