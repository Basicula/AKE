#include "Main/FractalExample.h"

#include "Common/ThreadPool.h"
#include "Fractal/JuliaSet.h"
#include "Fractal/MandelbrotSet.h"
#include "Fractal/MappingFunctions.h"
#include "Image/Image.h"
#include "Memory/custom_vector.h"
#include "Rendering/ImageRenderer.h"
#include "Window/EventListner.h"
#include "Window/GLUTWindow.h"
#include "Window/KeyboardEvent.h"

#include <memory>

namespace {
  class FractalChangeEventListner : public EventListner
  {
  public:
    FractalChangeEventListner(Fractal* ip_fractal)
      : mp_fractal(ip_fractal)
    {}

    void PollEvents() override {}

  protected:
    void _ProcessEvent(const Event& i_event) override
    {
      if (i_event.Type() != Event::EventType::KEY_PRESSED_EVENT)
        return;
      const auto& key_pressed_event = static_cast<const KeyPressedEvent&>(i_event);
      switch (key_pressed_event.Key()) {
        case KeyboardButton::KEY_W:
          origin_y += delta / scale;
          break;
        case KeyboardButton::KEY_S:
          origin_y -= delta / scale;
          break;
        case KeyboardButton::KEY_D:
          origin_x += delta / scale;
          break;
        case KeyboardButton::KEY_A:
          origin_x -= delta / scale;
          break;
        case KeyboardButton::KEY_Q:
          scale *= 1.5;
          break;
        case KeyboardButton::KEY_E:
          scale /= 1.5;
          break;
        default:
          break;
      }
      mp_fractal->SetOrigin(origin_x, origin_y);
      mp_fractal->SetScale(scale);
    }

  private:
    Fractal* mp_fractal;
    const float delta = 0.1f;
    float origin_x = 0.0f, origin_y = 0.0f;
    float scale = 1.0f;
  };

  class FractalRenderer : public ImageRenderer
  {
  public:
    explicit FractalRenderer(const Fractal& i_fractal, custom_vector<Color> i_color_map)
      : ImageRenderer()
      , m_fractal(i_fractal)
      , m_color_map(std::move(i_color_map))
    {}

  private:
    void _GenerateFrameImage() override
    {
      Parallel::ThreadPool::GetInstance()->ParallelFor(
        static_cast<std::size_t>(0), mp_image_source->GetSize(), [&](std::size_t i) {
          const int x = static_cast<int>(i % mp_image_source->GetWidth());
          const int y = static_cast<int>(i / mp_image_source->GetWidth());

          mp_image_source->SetPixel(
            x, y, static_cast<std::uint32_t>(FractalMapping::Default(m_fractal.GetValue(x, y), m_color_map)));
        });
    }

  private:
    const Fractal& m_fractal;
    custom_vector<Color> m_color_map;
  };
}

void FractalExample()
{
  const std::size_t width = 1024;
  const std::size_t height = 768;
  std::size_t max_iterations = 100;
  // std::unique_ptr<Fractal> p_fractal = std::make_unique<MandelbrotSet>(width, height, max_iterations);
  std::unique_ptr<Fractal> p_fractal = std::make_unique<JuliaSet>(width, height, max_iterations);
  custom_vector<Color> color_map{
    Color(0, 0, 0),       Color(66, 45, 15),    Color(25, 7, 25),    Color(10, 0, 45),    Color(5, 5, 73),
    Color(0, 7, 99),      Color(12, 43, 137),   Color(22, 81, 175),  Color(56, 124, 209), Color(132, 181, 229),
    Color(209, 234, 247), Color(239, 232, 191), Color(247, 201, 94), Color(255, 170, 0),  Color(204, 127, 0),
    Color(153, 86, 0),    Color(104, 51, 2),
  };
  // std::vector<Color> color_map
  //  {
  //  Color(0, 0, 0),
  //  Color(0, 0, 255),
  //  Color(0, 255, 0),
  //  Color(255, 0, 0),
  //  };
  // julia_set.SetColorMap(std::make_unique<SmoothColorMap>(color_map));
  // julia_set.SetType(JuliaSet::JuliaSetType::WhiskeryDragon);
  GLUTWindow window(width, height, "FracralsTest");
  window.InitRenderer<FractalRenderer>(*p_fractal, color_map);
  window.InitEventListner<FractalChangeEventListner>(p_fractal.get());
  window.Open();
}
