#pragma once

constexpr double SMOOTHING_RADIUS        = 0.0457;
constexpr double SMOOTHING_RADIUS_SQR    = 0.00208849; // SMOOTHING_RADIUS * SMOOTHING_RADIUS
constexpr double PARTICLE_MASS           = 0.02; // in kg
constexpr double GAS_STIFFNESS           = 3.0;
constexpr double WATER_DENSITY           = 1000.0; // kg/m^3 is rest density of water particle
constexpr double VISCOSITY               = 0.00089;
constexpr double WATER_SPEED_OF_SOUND    = 1531.0; // m / s