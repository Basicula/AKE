# .cpp or .c files
set(
  SOURCES

)

# .h or .hpp files
set(
  HEADERS
)

# optional if you have files that need to be compiled by nvcc
set(
  CUDA_FILES
)

# optional
set(
  PY
  
)

# optional
set(
  TESTS
  
)

# list of linking libs
set(
  LINKS
)

# list of linking libs that will be shared further up the hierarchy
set(
  PUBLIC_LINKS
)

generate_project(
  ProjectType(STATIC, SHARED etc)
  SUPPORT_CUDA # enabling cuda for project
  CUDA_FILES   ${CUDA_FILES}
  SOURCES      ${SOURCES}
  HEADERS      ${HEADERS}
  PY           ${PY}
  TESTS        ${TESTS}
  LINK         ${LINKS}
  PUBLIC_LINK  ${PUBLIC_LINKS}
)