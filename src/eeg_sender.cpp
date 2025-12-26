#include <arpa/inet.h>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <sys/socket.h>
#include <unistd.h>

#include "utils.h"

namespace {
bool send_all(int fd, const void* buffer, size_t len) {
    const char* buf = static_cast<const char*>(buffer);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = ::send(fd, buf, remaining, 0);
        if (n <= 0) {
            return false;
        }
        buf += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
}

void print_usage() {
    std::cout << "Usage: ./eeg_sender --host <ip> --port <port> --data_dir <dir>"
              << " [--max-samples <N>] [--interval-ms <ms>]" << std::endl;
}
}

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 5001;
    std::string data_dir = "data/eeg_data";
    size_t max_samples = 10;
    int interval_ms = 1000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) {
            if (i + 1 >= argc) {
                std::cerr << name << " requires a value" << std::endl;
                print_usage();
                return false;
            }
            return true;
        };

        if (arg == "--host") {
            if (!need_value("--host")) return 1;
            host = argv[++i];
        } else if (arg == "--port") {
            if (!need_value("--port")) return 1;
            port = std::stoi(argv[++i]);
        } else if (arg == "--data_dir") {
            if (!need_value("--data_dir")) return 1;
            data_dir = argv[++i];
        } else if (arg == "--max-samples") {
            if (!need_value("--max-samples")) return 1;
            max_samples = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--interval-ms") {
            if (!need_value("--interval-ms")) return 1;
            interval_ms = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else {
            std::cerr << "Unknown arg: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    int sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid host: " << host << std::endl;
        ::close(sockfd);
        return 1;
    }

    if (::connect(sockfd, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        std::cerr << "Failed to connect to " << host << ":" << port << std::endl;
        ::close(sockfd);
        return 1;
    }

    std::vector<std::string> files;
    for (auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());

    size_t sent = 0;
    for (const auto& path : files) {
        if (max_samples > 0 && sent >= max_samples) break;
        try {
            auto sample = load_npz_2d_array_float(path, "data");
            int32_t rows = static_cast<int32_t>(sample.size());
            int32_t cols = rows > 0 ? static_cast<int32_t>(sample[0].size()) : 0;
            if (rows <= 0 || cols <= 0) {
                std::cerr << "Skip invalid sample: " << path << std::endl;
                continue;
            }

            std::vector<float> flat;
            flat.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            for (const auto& row : sample) {
                if (static_cast<int32_t>(row.size()) != cols) {
                    std::cerr << "Inconsistent row size in sample: " << path << std::endl;
                    flat.clear();
                    break;
                }
                flat.insert(flat.end(), row.begin(), row.end());
            }
            if (flat.empty()) {
                continue;
            }

            int32_t rows_net = htonl(rows);
            int32_t cols_net = htonl(cols);
            if (!send_all(sockfd, &rows_net, sizeof(rows_net)) ||
                !send_all(sockfd, &cols_net, sizeof(cols_net)) ||
                !send_all(sockfd, flat.data(), flat.size() * sizeof(float))) {
                std::cerr << "Failed to send sample: " << path << std::endl;
                break;
            }

            ++sent;
            std::cout << "Sent sample " << sent << "/" << max_samples << ": " << path << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "Failed to load " << path << ": " << ex.what() << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    int32_t zero = 0;
    send_all(sockfd, &zero, sizeof(zero));
    send_all(sockfd, &zero, sizeof(zero));
    std::cout << "Sent termination signal." << std::endl;

    ::close(sockfd);
    return 0;
}
