#pragma once
/*
|------------------------------------------------------------------------------|
|                    SI table                                                  |
|------------------------------------------------------------------------------|
|      Full name      | Short name |                Description                |
|---------------------|------------|-------------------------------------------|
|Kilogram             |    (kg)    | Unit of mass                              |
|Metre                |     (m)    | Unit of length                            |
|Second               |     (s)    | Unit of time                              |
|                     |            |                                           |
|Area                 |    (m^2)   | Square metre, unit of surface area        |
|Volume               |    (m^3)   | Cubic metre, unit of boundary/solid volume|
|Speed, Velocity      |    (m/s)   | Metre per second, unit of speed           |
|Acceleration         |   (m/s^2)  | Metre per second squared                  |
|Density, Mass density|   (kg/m^3) | Kilogram per cubic metre                  |
|------------------------------------------------------------------------------|
*/

// All constants must be written in SI
namespace Physics::Constants {
  constexpr double DEFAULT_TIME_STEP = 0.05;
  constexpr double GRAVITY_CONSTANT = 9.81;           // m / s^2
  constexpr double GRAVITATIONAL_CONSTANT = 6.67e-11; // Newtons * kg^-2 * m^2
}
