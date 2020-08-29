# OrderedThreadPool

This is a lightweight implementation of a threadpool with synchronization.

## Quick Example
Imagine you want to parallelize the following code -

```c++
while (...) {
  std::cout << CostlyFn(input)) << std::endl;
}
```

Also imagine that you want outputs to be printed in order.

With OrderedThreadPool, you can achieve it like below -

```c++
OrderedThredPool<std::string> pool{10, 1};
while (...) {
  pool.Do(
    [&input] { return CostlyFn(input); },
    [](const std::string& out) {
      std::cout << out << std::endl;
    });
}
```

Note how this reads very similar to the original code. This does a few things -
* Spawns 10 threads and distributes incoming tasks.
* If `CostlyFn(...)` takes too long and all threads are busy, this blocks further input. This is governed by the second construction parameter, and can be turned off by passing 0.
* Even though the function will now be parallelized, the outputs will be displayed in order.

## Features

* Avoids thread spawn overhead by instantiating and maintaining threads on construction.
* Allows disabiling all threading by setting number of threads to 0. This allows rapid prototyping, and can help in debugging threaded logic in a client.
* Accepts a limit to throttle main thread if queue gets filled up faster than the jobs are finished.

# Dependencies

The libraries are header-only, and none of the following dependencies are required to use them in your project.

Nonetheless, they are required to build the tests.

## Ninja

Ninja is a fast build system alternative to make.

Install with -
```bash
sudo apt install ninja-build
```

## GoogleTest

This uses GoogleTest for unit testing. Following commands install it in Debian distros -

```bash
sudo apt install libgtest-dev build-essential cmake
mkdir gtest_build && cd gtest_build
cmake /usr/src/googletest
sudo cmake --build . --target install
cd .. && rm -r gtest_build
```
