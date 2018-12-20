project(KidnappedVehicle)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources ./main.cpp ./helper_functions.h ./json.hpp ./map.h ./particle_filter.cpp  ./particle_filter.h )


add_executable(KidnappedVeh ${sources})

target_link_libraries(KidnappedVeh z ssl uv uWS)
