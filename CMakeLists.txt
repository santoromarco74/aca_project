# ===========================================================================
#  Build del progetto K-means (ACA).
#
#  CUDA e OpenMP sono entrambi opzionali e rilevati automaticamente: il
#  progetto compila e i test girano anche su una macchina senza GPU, dove
#  vengono costruite le sole versioni seriale e OpenMP.
#
#    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#    cmake --build build -j
#    ./build/kmeans_test
#    ./build/kmeans_bench --n 200000 --d 32 --k 16
# ===========================================================================

cmake_minimum_required(VERSION 3.18)
project(aca_kmeans LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo di build" FORCE)
endif()

# -O3 con auto-vettorizzazione: la baseline seriale deve essere ottimizzata,
# altrimenti lo speedup misurato e' gonfiato e l'analisi perde significato.
# Niente -ffast-math: cambierebbe la semantica in virgola mobile e renderebbe
# il confronto numerico con le versioni CUDA non interpretabile.
if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

set(KMEANS_SOURCES
    src/common.cpp
    src/kmeans_serial.cpp
    src/kmeans_omp.cpp
)

# --- OpenMP ---------------------------------------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP trovato: la versione multicore verra' compilata")
else()
    message(WARNING "OpenMP non trovato: kmeans_omp degradera' alla versione seriale")
endif()

# --- CUDA (opzionale) -----------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # sm_60 e' il minimo richiesto: la riduzione dell'SSE usa atomicAdd su
    # double, disponibile solo da Pascal in avanti.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
    endif()
    list(APPEND KMEANS_SOURCES src/kmeans_cuda.cu)
    message(STATUS "CUDA trovato: architetture ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA non trovato: verranno costruite solo le versioni CPU")
endif()

add_library(kmeans_core STATIC ${KMEANS_SOURCES})
target_include_directories(kmeans_core PUBLIC include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kmeans_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(kmeans_core PUBLIC USE_CUDA)
    set_target_properties(kmeans_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)
    # -lineinfo: indispensabile per far correlare a Nsight Compute i contatori
    # hardware con le righe di sorgente.
    target_compile_options(kmeans_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --use_fast_math=false>)
endif()

add_executable(kmeans_bench src/main.cpp)
target_link_libraries(kmeans_bench PRIVATE kmeans_core)

add_executable(kmeans_test tests/test_correctness.cpp)
target_link_libraries(kmeans_test PRIVATE kmeans_core)

enable_testing()
add_test(NAME correctness COMMAND kmeans_test)
