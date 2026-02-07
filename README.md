Assignment 1 – Single-Core Matrix Multiplication

This repository contains implementations and results for multiple exercises related to single-core matrix multiplication and terminal command usage.

Repository Structure

.
├── Exercise 1/
│ ├── Lab1_TiledMatmul_CPU.cpp
│ └── out
├── Exercise 7/
│ ├── CMakeLists.txt
│ ├── lab7_matmul.cpp
│ └── kernels/
│ ├── compute/
│ │ └── matmul_tiles.cpp
│ └── dataflow/
│ ├── read_matmul_tiles.cpp
│ └── write_matmul_tiles.cpp
├── Terminal commands exercise 2-6/
│ ├── exercise 2.txt
│ ├── exercise 3.txt
│ ├── exercise 4.txt
│ ├── exercise 5 Terminal A.txt
│ ├── exercise 5 Terminal B.txt
│ └── exercise 6.txt
└── screen shots/
├── Exercise 1 results.png
└── Exercise 7 results.png

Exercise 1/: Source code and output for Exercise 1 (CPU tiled matrix multiplication)

Exercise 7/: Source code and build files for Exercise 7

Terminal commands exercise 2-6/: Text files containing terminal commands and outputs for Exercises 2–6

screen shots/: Screenshots of results for Exercises 1 and 7

Requirements

C++ compiler with C++11 or newer support (e.g. g++, clang++)

CMake (for Exercise 7)

Unix-based environment (macOS or Linux)

Exercise 1: Tiled Matrix Multiplication (CPU)

Build and Run

Navigate to the Exercise 1 directory:
cd "Exercise 1"

Compile the program:
g++ -O2 Lab1_TiledMatmul_CPU.cpp -o matmul

Run the executable:
./matmul

Output and results will be printed to the terminal.
Screenshots of the results are available in the screen shots/ directory.

Exercise 7: Matrix Multiplication with CMake

Build and Run

Navigate to the Exercise 7 directory:
cd "Exercise 7"

Create a build directory:
mkdir build
cd build

Configure the project with CMake:
cmake ..

Build the project:
make

Run the executable:
./lab7_matmul

Output will be displayed in the terminal.
Corresponding screenshots are provided in the screen shots/ directory.

Exercises 2–6: Terminal Commands

Navigate to:
cd "Terminal commands exercise 2-6"

Each .txt file contains the commands executed and their outputs for the corresponding exercise.

Notes

All experiments were run on a single core

Screenshots are included to verify correctness and output consistency

Build steps were tested on macOS
