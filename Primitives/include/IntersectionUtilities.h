#pragma once

#include <Sphere.h>
#include <Ray.h>
#include <Plane.h>

#include <algorithm>

bool IntersectRayWithSphere(Vector3d& o_intersection, const Ray& i_ray, const Sphere& i_sphere);

bool IntersectRayWithPlane(Vector3d& o_intersection, const Ray& i_ray, const Plane& i_plane);