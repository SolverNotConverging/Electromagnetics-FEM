# One version for root builds and standalone application builds.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../pyproject.toml" FEM_VERSION_LINE
    REGEX "^version = ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" FEM_VERSION "${FEM_VERSION_LINE}")
if(NOT FEM_VERSION)
    message(FATAL_ERROR "Cannot read the FEM version from pyproject.toml")
endif()
