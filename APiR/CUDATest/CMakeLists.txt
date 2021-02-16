set(
  SOURCES
  
  main.cu
)

generate_project(
  EXECUTABLE
  SUPPORT_CUDA
  SOURCES ${SOURCES}
  LINK    CUDACore
          Rendering
            Visual
              Math
)