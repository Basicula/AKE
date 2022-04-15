#pragma once

class Simulation
{
public:
  Simulation();
  explicit Simulation(double i_time_step);
  virtual ~Simulation() = default;

  // get/set time step in seconds
  void SetTimeStep(double i_new_time_step);
  [[nodiscard]] double GetTimeStep() const;

  void Update();

protected:
  virtual void _PreProcessing() = 0;
  virtual void _Update() = 0;
  virtual void _PostProcessing() = 0;

private:
  double m_time_step;
};
