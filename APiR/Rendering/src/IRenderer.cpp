#include <Rendering/IRenderer.h>

void IRenderer::SetOutputImage(Image* iop_image) {
  mp_frame_image = iop_image;
  _OutputImageWasSet();
  }

inline std::size_t IRenderer::GetRenderingDepth() const {
  return m_depth;
  }

inline void IRenderer::SetRenderingDepth(std::size_t i_depth) {
  m_depth = i_depth;
  }