function(source_groups)
  foreach(f IN ITEMS ${ARGV})
    get_filename_component(g ${f} DIRECTORY)
    if(g)
      string(REGEX REPLACE "/" "\\\\" g "${g}")
      source_group(${g} FILES "${f}")
    endif()
  endforeach()
endfunction()

# supposed file locations per subfolder:
#   headers                               - "include/${PROJECT_NAME}"
#   sources                               - "src"
#   py                                    - "py"
#   tests                                 - "test/c++"
#   precompile headers(pch.h and pch.cpp) - "src" 
function(generate_project)
  set (KEYWORDS
    SOURCES
    HEADERS
    PY
    TESTS
    LINK
  )
  cmake_parse_arguments(ARG "STATIC;SHARED;EXECUTABLE;SUPPORT_MFC;ENABLE_PCH;WIN32_EXE" "" "${KEYWORDS}" ${ARGN})
  get_filename_component(NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  
  if (ARG_ENABLE_PCH)
    set(PCH_FILES src/pch.h src/pch.cpp)
  endif()
  
  if (ARG_SUPPORT_MFC)
    add_definitions(-D_AFXDLL)
    set(CMAKE_MFC_FLAG 2)
  endif()
  
  set(FILES)
  foreach(FILE ${ARG_HEADERS})
    list(APPEND FILES "include/${NAME}/${FILE}")
  endforeach()
  foreach(FILE ${ARG_SOURCES})
    list(APPEND FILES "src/${FILE}")
  endforeach()
  
  if (ARG_SHARED)
    add_library(
      ${NAME}
      SHARED
      ${FILES}
      ${PCH_FILES}
    )
  elseif(ARG_EXECUTABLE)
    add_executable(
      ${NAME}
      ${FILES}
      ${PCH_FILES}
    )
  elseif (ARG_WIN32_EXE)
    add_executable(
      ${NAME}
      WIN32
      ${FILES}
      ${PCH_FILES}
    )
   SET_TARGET_PROPERTIES(${NAME} PROPERTIES LINK_FLAGS "/ENTRY:wWinMainCRTStartup")
  elseif(ARG_STATIC)
    add_library(
      ${NAME}
      STATIC
      ${FILES}
      ${PCH_FILES}
    )
  else()
    message(FATAL_ERROR "Project type doesn't specified, please set it to STATIC, SHARED or EXECUTABLE")
  endif()
  
  source_groups(${FILES})
  if(ARG_ENABLE_PCH)
    source_group("config" FILES ${PCH_FILES})
    add_pch(${NAME} ${PCH_FILES} ${FILES})
  endif()
  
  target_include_directories(${NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(${NAME} PRIVATE ${ARG_LINK})
  
  set_property(TARGET ${NAME} PROPERTY FOLDER "APiR")
endfunction()