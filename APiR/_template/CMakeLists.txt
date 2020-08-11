set(
  SOURCES
  
)

set(
  HEADERS
  
)

# optional
set(
  PY
  
)

# optional
set(
  TESTS
  
)

generatre_project(
  ProjectType(STATIC, SHARED etc)
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  PY      ${PY}
  TESTS   ${TESTS}
  LINK    ListOfLinkingLibs  
)