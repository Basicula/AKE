include(CMakeParseArguments)

function(source_groups)
  foreach(f IN ITEMS ${ARGV})
    get_filename_component(g ${f} DIRECTORY)
    if(g)
      string(REGEX REPLACE "/" "\\\\" g "${g}")
      source_group(${g} FILES "${f}")
    endif()
  endforeach()
endfunction()

function(add_pch tgt pchH pchCpp)
  set( pchObj "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${tgt}.pch" )
  
  if (MSVC)      
    set_property(TARGET ${tgt} APPEND_STRING PROPERTY COMPILE_FLAGS
        " /Fp\"${pchObj}\" /FI\"${pchH}\"")
        
    set_source_files_properties( ${pchCpp} PROPERTIES
      COMPILE_FLAGS "/Yc\"${pchH}\" /Fp\"${pchObj}\""
      OBJECT_OUTPUTS "${pchObj}" )
    
    set_source_files_properties( ${ARGN} PROPERTIES
      COMPILE_FLAGS "/Yu\"${pchH}\" /Fp\"${pchObj}\"" 
      OBJECT_DEPENDS "${pchObj}" )
          
  endif(MSVC)
endfunction()

# supposed file locations per subfolder:
#   headers                               - "include/${PROJECT_NAME}"
#   sources                               - "src"
#   py                                    - "py"
#   tests                                 - "tests/c++"
#   precompile headers(pch.h and pch.cpp) - "src" 
function(generate_project)
  set (KEYWORDS
    CUDA_FILES
    SOURCES
    HEADERS
    PY
    TESTS
    LINK
    PUBLIC_LINK
  )
  cmake_parse_arguments(ARG "STATIC;SHARED;EXECUTABLE;SUPPORT_MFC;ENABLE_PCH;WIN32_EXE;SUPPORT_CUDA" "" "${KEYWORDS}" ${ARGN})
  get_filename_component(NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  
  set(PCH_FILES)
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
  if (DEFINED ARG_PY)
    foreach(FILE ${ARG_PY})
      list(APPEND FILES "py/${FILE}")
    endforeach()
  endif()
  
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
    set_target_properties(${NAME} PROPERTIES LINK_FLAGS "/ENTRY:wWinMainCRTStartup")
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
  source_group("config" FILES ${PCH_FILES})
  if(ARG_ENABLE_PCH)
    add_pch(${NAME} ${PCH_FILES} ${FILES})
  endif()
  
  if(ARG_SUPPORT_CUDA AND ENABLE_CUDA)
    set_target_properties(${NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${ARG_CUDA_FILES} PROPERTIES LANGUAGE CUDA)
  endif()
  
  target_include_directories(${NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_include_directories(${NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/py")
  target_link_libraries(${NAME} PRIVATE ${ARG_LINK})
  target_link_libraries(${NAME} PUBLIC ${ARG_PUBLIC_LINK})
  
  set_property(TARGET ${NAME} PROPERTY FOLDER "APiR")
  
  if (DEFINED ARG_TESTS)
    set(TEST_FILES)
    foreach(FILE ${ARG_TESTS})
      list(APPEND TEST_FILES "tests/c++/${FILE}")
    endforeach()
    add_executable(
      ${NAME}.Tests
      ${TEST_FILES}
    )
    target_link_libraries(${NAME}.Tests PRIVATE ${NAME})
    target_link_libraries(${NAME}.Tests PRIVATE ${ARG_LINK})
    target_link_libraries(${NAME}.Tests PUBLIC  ${ARG_PUBLIC_LINK})
    target_link_libraries(${NAME}.Tests PRIVATE gtest_main)
    add_test(
      NAME
      unit
      COMMAND
      ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${NAME}.Tests
    )
    
    set_property(TARGET ${NAME}.Tests PROPERTY FOLDER "APiR")
  endif()
endfunction()