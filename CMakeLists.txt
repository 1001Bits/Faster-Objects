cmake_minimum_required(VERSION 3.23)

set(NAME "SkyrimPrecombines")
set(VERSION 1.0.0)

message(">>> Start build for '${NAME}' v${VERSION}")

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SOURCE_DIR "${ROOT_DIR}/src")
set(BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR ">> In-source builds not allowed.")
endif()

# ── vcpkg setup (BEFORE project()) ──────────────────────────────────────────
macro(set_from_environment VARIABLE)
  if(NOT DEFINED ${VARIABLE} AND DEFINED ENV{${VARIABLE}})
    set(${VARIABLE} $ENV{${VARIABLE}})
  endif()
endmacro()

set_from_environment(VCPKG_ROOT)
if(NOT DEFINED VCPKG_ROOT AND DEFINED CMAKE_TOOLCHAIN_FILE)
  get_filename_component(VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}/../../.." ABSOLUTE)
endif()
if(DEFINED VCPKG_ROOT)
  set(VCPKG_ROOT "${VCPKG_ROOT}" CACHE PATH "vcpkg checkout")
  set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  set(VCPKG_TARGET_TRIPLET "x64-windows-static-md" CACHE STRING "")
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "")
else()
  message(FATAL_ERROR ">> Variable VCPKG_ROOT is not set!")
endif()

project(${NAME} VERSION ${VERSION} LANGUAGES CXX)
message(">>> Building project '${PROJECT_NAME}' v${PROJECT_VERSION}")

# ── Compiler settings ────────────────────────────────────────────────────────
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)

# ── CommonLibSSE-NG v13.0.0 (multi-runtime: SE 1.5.97, AE incl. 1.7.x, VR) ────
# GPL-3.0-or-later with the Modding and Linking exceptions. Keep the immutable
# commit pin in addition to the version so a moved or recreated tag cannot
# change builds.
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(COMMONLIBSSE_NG_VERSION "13.0.0")
set(COMMONLIBSSE_NG_COMMIT "bf27825961c932c940304317156c412cda8300a7")
set(COMMONLIBSSE_OPENVR_COMMIT "60eb187801956ad277f1cae6680e3a410ee0873b")
set(COMMONLIBSSE_OPENVR_HEADER_SHA256
    "0ffc25d09ddb50c78a56f15cc48fc724c55d1f4d06fd2ca009c9f985158e7f8c")
set(COMMONLIBSSE_OPENVR_LIBRARY_SHA256
    "b92394fb290d9a99cae64af5ef5a71bf723dea4640f513acb9e43eeaa86cbcf5")
set(BUILD_TESTS OFF CACHE BOOL "Do not build CommonLibSSE tests" FORCE)
set(ENABLE_SKYRIM_SE ON CACHE BOOL "CommonLibSSE: support Skyrim SE" FORCE)
set(ENABLE_SKYRIM_AE ON CACHE BOOL "CommonLibSSE: support Skyrim AE" FORCE)
set(ENABLE_SKYRIM_VR ON CACHE BOOL "CommonLibSSE: support Skyrim VR" FORCE)
set(SKSE_SUPPORT_XBYAK OFF CACHE BOOL "CommonLibSSE: no Xbyak trampoline API" FORCE)
set(REX_OPTION_INI OFF CACHE BOOL "CommonLibSSE: no REX INI settings" FORCE)
set(REX_OPTION_JSON OFF CACHE BOOL "CommonLibSSE: no REX JSON settings" FORCE)
set(REX_OPTION_TOML OFF CACHE BOOL "CommonLibSSE: no REX TOML settings" FORCE)
# Instruction-boundary check on trampoline patch sites; logs only (debug level), never blocks.
set(SKSE_SUPPORT_PATCH_SAFETY ON CACHE BOOL "CommonLibSSE: patch-site boundary check" FORCE)
# Always build CommonLib from the pinned source; never download a prebuilt library.
set(COMMONLIB_PREBUILT OFF CACHE BOOL "CommonLibSSE: no prebuilt bundle" FORCE)
set(COMMONLIB_PREBUILT_MULTICONFIG OFF CACHE BOOL "CommonLibSSE: no prebuilt bundle" FORCE)
unset(COMMONLIB_PREBUILT_DIR CACHE)

find_package(Git REQUIRED)
FetchContent_Declare(
    CommonLibSSE
    GIT_REPOSITORY https://github.com/alandtse/CommonLibSSE-NG.git
    GIT_TAG "v${COMMONLIBSSE_NG_VERSION}"
    GIT_SHALLOW TRUE
    GIT_SUBMODULES ""
    SOURCE_SUBDIR _validate_pin_before_add_subdirectory
)
# The patch-safety check vendors MinHook's hde64 through FetchContent_Declare(hde64 ... GIT_TAG v1.3.4).
# v1.3.4 is commit c3fcafdc, pinned here by archive hash; the first declaration of a name wins, so this one
# replaces CommonLib's unpinned git fetch. SOURCE_SUBDIR has no CMakeLists.txt, so nothing is added as a
# subdirectory; CommonLib compiles src/hde/hde64.c itself.
FetchContent_Declare(
    hde64
    URL "https://codeload.github.com/TsudaKageyu/minhook/zip/c3fcafdc10146beb5919319d0683e44e3c30d537"
    URL_HASH SHA256=cdcb160f734d81bd4d235dfea79e3f5a661c8ef0ab74fa814272aa5449069034
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR src/hde
)
FetchContent_MakeAvailable(CommonLibSSE)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -c "safe.directory=${commonlibsse_SOURCE_DIR}" -C "${commonlibsse_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE COMMONLIBSSE_PIN_RESULT
    OUTPUT_VARIABLE COMMONLIBSSE_RESOLVED_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT COMMONLIBSSE_PIN_RESULT EQUAL 0 OR
   NOT COMMONLIBSSE_RESOLVED_COMMIT STREQUAL COMMONLIBSSE_NG_COMMIT)
    message(FATAL_ERROR
        "CommonLibSSE-NG v${COMMONLIBSSE_NG_VERSION} did not resolve to the expected commit "
        "${COMMONLIBSSE_NG_COMMIT} (resolved: ${COMMONLIBSSE_RESOLVED_COMMIT})")
endif()

# CommonLibSSE only consumes openvr.h and the Win64 import library. Fetch those
# exact, content-hashed artifacts instead of cloning OpenVR's large repository.
set(COMMONLIBSSE_OPENVR_ROOT "${commonlibsse_SOURCE_DIR}/extern/openvr")
file(MAKE_DIRECTORY
    "${COMMONLIBSSE_OPENVR_ROOT}/headers"
    "${COMMONLIBSSE_OPENVR_ROOT}/lib/win64"
)
file(DOWNLOAD
    "https://raw.githubusercontent.com/ValveSoftware/openvr/${COMMONLIBSSE_OPENVR_COMMIT}/headers/openvr.h"
    "${COMMONLIBSSE_OPENVR_ROOT}/headers/openvr.h"
    EXPECTED_HASH "SHA256=${COMMONLIBSSE_OPENVR_HEADER_SHA256}"
    TLS_VERIFY ON
)
file(DOWNLOAD
    "https://raw.githubusercontent.com/ValveSoftware/openvr/${COMMONLIBSSE_OPENVR_COMMIT}/lib/win64/openvr_api.lib"
    "${COMMONLIBSSE_OPENVR_ROOT}/lib/win64/openvr_api.lib"
    EXPECTED_HASH "SHA256=${COMMONLIBSSE_OPENVR_LIBRARY_SHA256}"
    TLS_VERIFY ON
)

add_subdirectory("${commonlibsse_SOURCE_DIR}" "${commonlibsse_BINARY_DIR}")
include("${commonlibsse_SOURCE_DIR}/cmake/CommonLibSSE.cmake")
# The pinned dependency uses a generator expression in the boolean IPO property.
# Set the configuration-specific properties explicitly so reconfiguration cannot
# change its Release code-generation mode after check_ipo_supported succeeds.
get_target_property(COMMONLIBSSE_IS_IMPORTED CommonLibSSE IMPORTED)
if(NOT COMMONLIBSSE_IS_IMPORTED)
  set_target_properties(CommonLibSSE PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
      INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)
  if(MSVC)
    target_compile_options(CommonLibSSE PRIVATE /MP4)
  endif()
endif()

# ── Additional dependencies ─────────────────────────────────────────────────
find_package(spdlog REQUIRED CONFIG)
find_package(nlohmann_json REQUIRED CONFIG)

# ── Sources ──────────────────────────────────────────────────────────────────
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${SOURCE_DIR}/*.cpp"
    "${SOURCE_DIR}/*.h"
)

# The runtime bake is part of the plugin: its sources, and the merge core linked in below (PrecombineMeshCore).
# BakePlugin.cpp's own SKSE entry point is compiled only with PRECOMBINES_BAKE_PLUGIN.
file(GLOB RUNTIME_BAKE_SOURCES CONFIGURE_DEPENDS "${ROOT_DIR}/runtime_bake/*.cpp" "${ROOT_DIR}/runtime_bake/*.h")

# add_commonlibsse_plugin handles all the multi-runtime magic
add_commonlibsse_plugin(${PROJECT_NAME} SOURCES ${SOURCES} ${RUNTIME_BAKE_SOURCES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
target_precompile_headers(${PROJECT_NAME} PRIVATE "${SOURCE_DIR}/PCH.h")

target_include_directories(${PROJECT_NAME} PRIVATE
    "${SOURCE_DIR}"
    "${ROOT_DIR}/runtime_bake"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    UNICODE
    _UNICODE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

# ── MSVC-specific options ────────────────────────────────────────────────────
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    target_compile_options(${PROJECT_NAME} PRIVATE
        /MP4
        /W3
        /permissive-
        /Zc:preprocessor
        /Zc:__cplusplus
        /Zi
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        /DEBUG
        /OPT:REF
        /OPT:ICF
    )
endif()

FetchContent_Declare(nifly
    GIT_REPOSITORY https://github.com/ousnius/nifly.git
    GIT_TAG cca0a770094bb962fb28ea1fec5ea903e68fda8e
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(nifly)
execute_process(COMMAND "${GIT_EXECUTABLE}" -c "safe.directory=${nifly_SOURCE_DIR}"
    -C "${nifly_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE NIFLY_RESOLVED_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NIFLY_PIN_RESULT ERROR_QUIET)
if(NOT NIFLY_PIN_RESULT EQUAL 0 OR
   NOT NIFLY_RESOLVED_COMMIT STREQUAL "cca0a770094bb962fb28ea1fec5ea903e68fda8e")
  message(FATAL_ERROR "nifly does not match the pinned commit")
endif()
# The merge core is a library (precombine::Bake) the plugin bakes with in-process. nifly is GPL-3.0, as is this mod.
add_library(PrecombineMeshCore STATIC "${ROOT_DIR}/generator/PrecombineMeshCore.cpp")
target_compile_features(PrecombineMeshCore PUBLIC cxx_std_20)
target_include_directories(PrecombineMeshCore PUBLIC "${ROOT_DIR}/generator")
target_link_libraries(PrecombineMeshCore PUBLIC nifly nlohmann_json::nlohmann_json)
target_compile_definitions(PrecombineMeshCore PUBLIC NOMINMAX)
target_link_libraries(${PROJECT_NAME} PRIVATE PrecombineMeshCore)

message(">>> ${PROJECT_NAME} configured successfully")
