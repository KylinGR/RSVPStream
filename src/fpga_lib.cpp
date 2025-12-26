#include "fpga_lib.h"

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// 全局变量定义
std::vector<uint16_t> ram_coeff_b16;
int verbose = 0;
int fdC2H = -1, fdH2C = -1;
int fdReg = -1;
char *allocated = NULL;
uint8_t* map = NULL;
float X_global_scale[100];
float X_local_scale[100];
static size_t allocated_size = 0;

namespace {
std::filesystem::path dat_dir_path;
std::mutex dat_dir_mtx;

std::filesystem::path resolve_dat_dir() {
    const char* env = std::getenv("RSVP_DAT_DIR");
    if (env && *env) return std::filesystem::path(env);
    return std::filesystem::path("data/dat");
}
}

void set_dat_dir(const std::filesystem::path& dir) {
    std::lock_guard<std::mutex> lock(dat_dir_mtx);
    dat_dir_path = dir;
}

std::filesystem::path dat_dir() {
    std::lock_guard<std::mutex> lock(dat_dir_mtx);
    if (dat_dir_path.empty()) {
        dat_dir_path = resolve_dat_dir();
    }
    return dat_dir_path;
}

float hexToFloat_ptr(uint32_t hex) {
    return *reinterpret_cast<float*>(&hex);
}

uint32_t floatToHex(float f) {
    uint32_t hexValue;
    memcpy(&hexValue, &f, sizeof(float)); // 将浮点数的内存布局复制到整数
    return hexValue;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void load_datfile2coeff_b16(const std::string& strFilename,uint32_t reg_adr,uint16_t reg_len)
{
    std::filesystem::path strFile = dat_dir() / strFilename;
    uint64_t tot_size;
 	printf("strFile =%s\n",strFile.c_str());
    ram_coeff_b16.resize(reg_len); // ensure buffer is large enough for the read
    std::string strFileStr = strFile.string();
    file_base file(strFileStr,FILE_DIR::FIN,FILE_TYPE::BINARY);		

    // Read at most the expected number of bytes to avoid overflowing ram_coeff_b16.
    const uint32_t expected_bytes = static_cast<uint32_t>(reg_len) * static_cast<uint32_t>(sizeof(uint16_t));
    const int32_t read_bytes = file.read(reinterpret_cast<char*>(ram_coeff_b16.data()), expected_bytes);
    tot_size = (read_bytes > 0) ? static_cast<uint64_t>(read_bytes) : 0;
    uint32_t elems = std::min<uint32_t>(reg_len, static_cast<uint32_t>(tot_size / sizeof(uint16_t)));
    for(uint32_t i=0;i<elems;i++)
	{
		RegWr(reg_adr+i*4,ram_coeff_b16[i]);
	//	std::cout<<"ram_coeff["<<std::hex<<reg_adr+i*4<<"]="<<ram_coeff_b16[i]<<std::endl;
	}
    if (elems < reg_len) {
        fprintf(stderr, "%s size %lu bytes is smaller than expected %u bytes, wrote %u elements.\n",
                strFile.c_str(), tot_size, reg_len * (uint32_t)sizeof(uint16_t), elems);
    }
}


void load_coeff()
{
	load_datfile2coeff_b16("ptrim.dat",ADR_XORDER_PTRIM_RAM_L,492*2);
	load_datfile2coeff_b16("bglobal.dat",ADR_BG_RAM_L,12*2);//bglobal
	load_datfile2coeff_b16("betaglobal.dat",ADR_BETA_RAM_L,17*2);	//betaglobal
	load_datfile2coeff_b16("gstf_weight.dat",ADR_GST_RAM_L,12*2);//gsfweight
	load_datfile2coeff_b16("lr_model.dat",ADR_LR_MODEL_RAM_L,2544*2);//lrmodel
}

ssize_t write_from_buffer(char *fname, int fd, char *buffer, uint64_t size, uint64_t base)
{
	ssize_t rc;
	uint64_t count = 0;
	char *buf = buffer;
	off_t offset = base;
	int loop = 0;

	while (count < size) {
		uint64_t bytes = size - count;

		if (bytes > RW_MAX_SIZE)
			bytes = RW_MAX_SIZE;

		if (offset) {
			rc = lseek(fd, offset, SEEK_SET);
			if (rc != offset) {
				fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
					fname, rc, offset);
				perror("seek file");
				return -EIO;
			}
		}

		/* write data to file from memory buffer */
		rc = write(fd, buf, bytes);
		if (rc < 0) {
			fprintf(stderr, "%s, write 0x%lx @ 0x%lx failed %ld.\n",
				fname, bytes, offset, rc);
			perror("write file");
			return -EIO;
		}

		count += rc;
		if (rc != bytes) {
			fprintf(stderr, "%s, write underflow 0x%lx/0x%lx @ 0x%lx.\n",
				fname, rc, bytes, offset);
			break;
		}
		buf += bytes;
		offset += bytes;

		loop++;
	}	

	if (count != size && loop)
		fprintf(stderr, "%s, write underflow 0x%lx/0x%lx.\n",
			fname, count, size);

	return count;
}

void RegWr(uint32_t u32Adr, uint32_t u32Dat)
{
    DELAY_US(10);
    *((uint32_t *)(map+ u32Adr)) = u32Dat;
}

uint32_t RegRd(uint32_t u32Adr)
{
    DELAY_US(10);
    return *((uint32_t *) (map+ u32Adr));
}

void File2DDR(string& strFile, uint32_t ddr_sta_adr)
{
    // Stream file to FPGA DDR in chunks to avoid any host-side buffer overflow.
    const int in_fd = ::open(strFile.c_str(), O_RDONLY);
    if (in_fd < 0) {
        throw std::runtime_error("File2DDR: failed to open " + strFile + ": " + std::string(strerror(errno)));
    }

    // Ensure we have a staging buffer.
    if (allocated == NULL || allocated_size == 0) {
        const size_t alignment = 1048576; // 1MB
        const size_t init_size = HZHY_PS_BUF_LEN;
        if (posix_memalign((void **)&allocated, alignment, init_size) != 0 || allocated == NULL) {
            ::close(in_fd);
            throw std::runtime_error("File2DDR: posix_memalign failed");
        }
        allocated_size = init_size;
    }

    uint64_t offset = 0;
    while (true) {
        const size_t chunk = allocated_size;
        ssize_t n = ::read(in_fd, allocated, chunk);
        if (n == 0) {
            break; // EOF
        }
        if (n < 0) {
            const std::string err = strerror(errno);
            ::close(in_fd);
            throw std::runtime_error("File2DDR: read failed for " + strFile + ": " + err);
        }
        (void)write_from_buffer(FPGA_H2C_NODE, fdH2C, (char *)allocated, (uint64_t)n, ddr_sta_adr + offset);
        offset += (uint64_t)n;
    }

    ::close(in_fd);
}

void Vec2DDR(const std::vector<int16_t>& data, uint32_t ddr_sta_adr)
{
    size_t byte_size = data.size() * sizeof(int16_t);
    // 直接写数据到FPGA DDR
    int ret = write_from_buffer(FPGA_H2C_NODE, fdH2C, 
                                reinterpret_cast<char*>(const_cast<int16_t*>(data.data())), 
                                byte_size, ddr_sta_adr);
    // if (ret != 0) {
    //     fprintf(stderr, "Vec2DDR, addr=0x%x, size=%zu\n", ddr_sta_adr, byte_size);
    // }
}

void NetRegInit()
{
    RegWr(ADR_WR_M_LOCAL_SCALE, 0x454fcb4d);
    RegWr(ADR_GAMMA_LOCAL_SCALE, 0x471f987c);
    RegWr(ADR_BETA_LOCAL_SCALE, 0x473389d2);
    RegWr(ADR_B_LOCAL_SCALE, 0x47c8e2b8);
    RegWr(ADR_W_LOCAL_SCALE, 0x483040e4);
    RegWr(ADR_LR_MODEL_SCALE, 0x477ffe00);
    RegWr(ADR_GAMMA_GLOBSCALE, 0x478a28a4);
    RegWr(ADR_MGLOBAL_SCALE, 0x444cf3b8);
    RegWr(ADR_BETAGLOB_SCALE, 0x484bbac5);
    RegWr(ADR_WGLOBAL_SCALE, 0x489dacf6);
    RegWr(ADR_QGLOB_SCALE, 0x48bfc446);
    RegWr(ADR_BGLOBAL_SCALE, 0x484f719e);
    RegWr(ADR_GSTF_WEIGHT_SCALE, 0x47d553aa);
}

void Start()
{
    RegWr(ADR_ALG_START, 0x0);
    DELAY_MS(1);
    RegWr(ADR_ALG_START, 0x1);
}

void Reset()
{
    DELAY_MS(10);
    RegWr(ADR_LOGIC_RST, 0x0);
}

void LoadScale()
{
    std::string strFile;
    uint64_t tot_size;
    strFile = "/home/hzhy/cpp_work/dat/X_global_scale_all100.dat";
    printf("strFile =%s\n", strFile.c_str());
    file_base file(strFile, FILE_DIR::FIN, FILE_TYPE::BINARY);		
    tot_size = file.readAll((char*)X_global_scale);

    strFile = "/home/hzhy/cpp_work/dat/X_local_scale_all100.dat";
    printf("strFile =%s\n", strFile.c_str());
    file_base file1(strFile, FILE_DIR::FIN, FILE_TYPE::BINARY);		
    tot_size = file1.readAll((char*)X_local_scale);
}

int InitFPGA()
{
    fdReg = open("/dev/xdma0_user", O_RDWR|O_SYNC);
    if (fdReg < 0) {
        printf("Open register node /dev/xdma0_user failed: %s.\n", strerror(errno));
        return -1;
    }

    map = (uint8_t*)mmap(NULL, 131072, PROT_READ | PROT_WRITE, MAP_SHARED, fdReg, 0x0);
    if (map == (void *)-1) {
        printf("Register Memory  mapped failed: %s.\n", strerror(errno));
        close(fdReg);
        fdReg = -1;
        return -1;
    }

    /*将数据从主机写入到FPGA*/
    fdH2C = open(FPGA_H2C_NODE, O_RDWR);
    if(fdH2C < 0) {
        printf("Failed to open node %s!\n", FPGA_H2C_NODE);
        return -1;
    } else {        
        printf("Open node %s success\n", FPGA_H2C_NODE);
    }
    
    if (posix_memalign((void **)&allocated, 1048576 /*alignment */ , HZHY_PS_BUF_LEN) != 0 || allocated == NULL) {
        printf("posix_memalign failed\n");
        return -1;
    }
    allocated_size = HZHY_PS_BUF_LEN;
    printf("allocated'a address=%p\n", allocated);
    
    return 0;
}

void CleanupFPGA()
{
    if (allocated) {
        free(allocated);
        allocated = NULL;
        allocated_size = 0;
    }
    if (fdH2C >= 0) {
        close(fdH2C);
    }
    if (fdReg >= 0) {
        close(fdReg);
    }
    if (map && map != (void*)-1) {
        munmap(map, 131072);
        map = NULL;
    }
}
