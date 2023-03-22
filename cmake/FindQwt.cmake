# Find Qwt
# ~~~~~~~~
# Copyright (c) 2010, Tim Sutton <tim at linfiniti.com>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#
# Once run this will define:
#
# QWT_FOUND       = system has QWT lib
# QWT_LIBRARY     = full path to the QWT library
# QWT_INCLUDE_DIR = where to find headers
#

set(QWT_ROOT
   "${QWT_ROOT}"
   CACHE
   PATH "Qwt Root directory"
)

FIND_PATH(QWT_INCLUDE_DIR NAMES qwt.h
  PATHS
      "${QWT_ROOT}"
      "$ENV{QWT_ROOT}"
      /usr/include
      /usr/include/qt5
      /usr/local/include
      /usr/local/include/qt5
      "$ENV{LIB_DIR}/include"
      "$ENV{INCLUDE}"
  PATH_SUFFIXES include
)
set ( QWT_INCLUDE_DIRS ${QWT_INCLUDE_DIR} )

if (NOT QWT_ROOT AND QWT_INCLUDE_DIR)
    set (QWT_ROOT "${QWT_INCLUDE_DIR}/..")
endif()

find_library(QWT_LIBRARY_RELEASE
  NAMES qwt
  PATHS "${QWT_ROOT}"
  PATH_SUFFIXES lib
)
find_library(QWT_LIBRARY_DEBUG
  NAMES qwtd
  PATHS "${QWT_ROOT}"
  PATH_SUFFIXES lib
)

# show in ccmake interface
MARK_AS_ADVANCED(
  QWT_INCLUDE_DIR
  QWT_LIBRARY
  FOUND_QWT
)

# version
set ( _VERSION_FILE ${QWT_INCLUDE_DIR}/qwt_global.h )
if ( EXISTS ${_VERSION_FILE} )
  file ( STRINGS ${_VERSION_FILE} _VERSION_LINE REGEX "define[ ]+QWT_VERSION_STR" )
  if ( _VERSION_LINE )
    string ( REGEX REPLACE ".*define[ ]+QWT_VERSION_STR[ ]+\"([^\"]*)\".*" "\\1" QWT_VERSION_STRING "${_VERSION_LINE}" )
    set(QWT_VERSION ${QWT_VERSION_STRING})
  endif ()
endif ()
unset ( _VERSION_FILE )

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args( Qwt REQUIRED_VARS QWT_LIBRARY_RELEASE QWT_LIBRARY_DEBUG QWT_INCLUDE_DIR VERSION_VAR QWT_VERSION_STRING )

if (QWT_FOUND AND NOT TARGET Qwt::Qwt)
    add_library(Qwt::Qwt UNKNOWN IMPORTED)
    if (QWT_LIBRARY_RELEASE)
        set_property(TARGET Qwt::Qwt APPEND PROPERTY
            IMPORTED_CONFIGURATIONS RELEASE
        )
        set_target_properties(Qwt::Qwt PROPERTIES
            IMPORTED_LOCATION_RELEASE "${QWT_LIBRARY_RELEASE}"
        )
    endif()
    if (QWT_LIBRARY_DEBUG)
        set_property(TARGET Qwt::Qwt APPEND PROPERTY
            IMPORTED_CONFIGURATIONS DEBUG
        )
        set_target_properties(Qwt::Qwt PROPERTIES
            IMPORTED_LOCATION_DEBUG "${QWT_LIBRARY_DEBUG}"
        )
    endif()
    set_target_properties(Qwt::Qwt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${QWT_INCLUDE_DIRS}"
        DEFINE_SYMBOL -DQWT_DLL
      )
    target_compile_definitions(Qwt::Qwt INTERFACE -DQWT_DLL)
endif ()
