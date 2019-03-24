#pragma once

#include <Plane.h>
#include <Sphere.h>
#include <Ray.h>

#include <algorithm>

bool IntersectRayWithSphere(Vector3d& o_intersection, const Ray& i_ray, const Sphere& i_sphere);

bool IntersectRayWithPlane(Vector3d& o_intersection, const Ray& i_ray, const Plane& i_plane);

bool IntersectRayWithObject(Vector3d& o_intersection, const Ray& i_ray, const IObject* ip_object);

bool IntersectRayWithWave(Vector3d& o_intersection, const Ray& i_ray);