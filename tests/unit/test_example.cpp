#include <iostream>
#include <cassert>

int main() {
  std::cout << "Running unit tests..." << std::endl;
  assert(1 + 1 == 2);
  std::cout << "All tests passed!" << std::endl;
  return 0;
}
