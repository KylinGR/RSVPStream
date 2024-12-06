enable_testing()

# 添加单元测试
add_executable(test_unit tests/unit/test_example.cpp)
add_test(NAME test_unit COMMAND test_unit)

# 添加集成测试
add_executable(test_integration tests/integration/test_pipeline.cpp)
add_test(NAME test_integration COMMAND test_integration)
