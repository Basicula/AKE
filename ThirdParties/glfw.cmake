include(FetchContent)

FetchContent_Declare(glfw
GIT_REPOSITORY "https://github.com/glfw/glfw"
GIT_TAG "3.3.3"
GIT_SHALLOW ON)
FetchContent_MakeAvailable(glfw)