find_library(LIBNVT
  NAMES
  nvToolsExt
  PATHS
  ${CUDAToolkit_LIBRARY_DIR}
  ${CUDAToolkit_LIBRARY_ROOT}
  PATH_SUFFIXES
  lib lib64 libs
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nvToolsExt
  ""
  LIBNVT
)

mark_as_advanced(LIBNVT)
if(nvToolsExt_FOUND AND NOT TARGET nvToolsExt)
  add_library(nvToolsExt UNKNOWN IMPORTED)
  set_target_properties(nvToolsExt PROPERTIES 
    IMPORTED_LOCATION "${LIBNVT}")
endif()