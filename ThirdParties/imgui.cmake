set(
  SOURCES
  
  ${THIRD_PARTIES}/imgui/imgui.cpp
  ${THIRD_PARTIES}/imgui/imgui_draw.cpp
  ${THIRD_PARTIES}/imgui/imgui_tables.cpp
  ${THIRD_PARTIES}/imgui/imgui_widgets.cpp
  ${THIRD_PARTIES}/imgui/backends/imgui_impl_glfw.cpp
  ${THIRD_PARTIES}/imgui/backends/imgui_impl_glut.cpp
  ${THIRD_PARTIES}/imgui/backends/imgui_impl_opengl3.cpp
  # unneeded for now
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_allegro5.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_android.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_dx10.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_dx11.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_dx12.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_dx9.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_marmalade.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_opengl2.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_sdl.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_vulkan.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_wgpu.cpp
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_win32.cpp
)

set(
  HEADERS
  
  ${THIRD_PARTIES}/imgui/imgui.h
  ${THIRD_PARTIES}/imgui/imgui_internal.h
  ${THIRD_PARTIES}/imgui/imstb_textedit.h
  ${THIRD_PARTIES}/imgui/imstb_truetype.h
  ${THIRD_PARTIES}/imgui/backends/imgui_impl_glfw.h
  ${THIRD_PARTIES}/imgui/backends/imgui_impl_glut.h
  ${THIRD_PARTIES}/imgui/backends/imgui_impl_opengl3.h
  # unneeded for now
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_allegro5.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_android.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_dx10.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_dx11.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_dx12.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_dx9.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_marmalade.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_metal.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_opengl2.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_osx.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_sdl.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_vulkan.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_wgpu.h
  #${THIRD_PARTIES}/imgui/backends/imgui_impl_win32.h
)

add_library(imgui STATIC ${HEADERS} ${SOURCES})

source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

target_include_directories(imgui PUBLIC "${THIRD_PARTIES}/imgui")
target_link_libraries(imgui PUBLIC glfw ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})

set_property(TARGET imgui PROPERTY FOLDER "imgui")