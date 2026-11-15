# Helpers for viewfeat example executables and for copying the link pattern into your project.

function(viewfeat_deploy_shared_libs target)
  foreach(_dep IN ITEMS viewfeat tensorrtbuffer)
    if(TARGET ${_dep})
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:${_dep}>"
          "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Deploy $<TARGET_FILE_NAME:${_dep}> next to ${target}"
      )
    endif()
  endforeach()
endfunction()

# Link pattern:
#   target_link_libraries(<your_target> PRIVATE viewfeat::viewfeat)
#   target_include_directories(<your_target> PRIVATE ${VIEWFEAT_ROOT}/include)  # only if not using the imported target
function(viewfeat_add_example target)
  if(NOT TARGET viewfeat)
    message(FATAL_ERROR "viewfeat_add_example: viewfeat target is missing. Build the library first.")
  endif()

  add_executable(${target} "${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp")
  target_include_directories(${target} PRIVATE
    "${VIEWFEAT_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(${target} PRIVATE viewfeat::viewfeat ${OpenCV_LIBS})
  add_dependencies(${target} viewfeat)

  # RPATH=$ORIGIN + copy .so beside the binary: run without relying on LD_LIBRARY_PATH.
  set_target_properties(${target} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
  )
  target_link_options(${target} PRIVATE
    "LINKER:--disable-new-dtags,-rpath,$<TARGET_FILE_DIR:${target}>"
  )
  viewfeat_deploy_shared_libs(${target})
endfunction()
