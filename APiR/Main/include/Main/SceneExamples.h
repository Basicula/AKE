#pragma once
#include <Geometry/Sphere.h>
#include <Geometry/Cylinder.h>
#include <Geometry/Torus.h>

#include <Math/Constants.h>

#include <Rendering/RenderableObject.h>
#include <Rendering/Scene.h>

#include <Visual/ColorMaterial.h>
#include <Visual/SpotLight.h>



namespace ExampleScene {
  Scene OneSphere();
  Scene NineSpheres();
  Scene EmptyRoom();
  Scene RotatableTorus();
  Scene ComplexScene();
  }