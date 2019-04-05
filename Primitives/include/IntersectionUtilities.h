#pragma once

#include <Plane.h>
#include <Sphere.h>
#include <Ray.h>
#include <Torus.h>
#include <Cylinder.h>

#include <algorithm>

bool IntersectRayWithSphere(Vector3d& o_intersection, double &o_distance, const Ray& i_ray, const Sphere& i_sphere);

bool IntersectRayWithPlane(Vector3d& o_intersection, double &o_distance, const Ray& i_ray, const Plane& i_plane);

bool IntersectRayWithTorus(Vector3d& o_intersecion, double &o_distance, const Ray& i_ray, const Torus& i_torus);

bool IntersectRayWithCylinder(Vector3d& o_intersecion, double &o_distance, const Ray& i_ray, const Cylinder& i_torus);

bool IntersectRayWithObject(Vector3d& o_intersection, double &o_distance, const Ray& i_ray, const IObject* ip_object);

bool IntersectRayWithWave(Vector3d& o_intersection, double &o_distance, const Ray& i_ray);