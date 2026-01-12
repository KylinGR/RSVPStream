#include <iostream>

#include "SystemGlobal.h"
// #include "HzhyGadget.h"
// #include "dma_utils.h"
#include <memory>
#include <thread>
#include <cmath>
#include "file_base.h"

#define  FPGA_C2H_NODE  "/dev/xdma0_c2h_0"
#define  FPGA_H2C_NODE  "/dev/xdma0_h2c_0"

#define HZHY_PS_BUF_LEN    (0x200000) //1M字节

#define HZHY_FPGA_ADDR_BASE  (0x0000000) //偏移16M的位置
#define RW_MAX_SIZE	0x7ffff000


#define XLOCAL_ADR        0x10000  
#define LOCAL_COEFF_ADR   0x100000
#define BLOCAL_COEFF_ADR  0x400000
#define GLOBAL_COEFF_ADR  0x500000
#define QGLOB_COEFF_ADR   0x800000
#define XGLOB_ADR         0x900000

#define        ADR_LOGIC_RST                      (0x0 <<2) 
#define        ADR_ALG_START                      (0x1 <<2) 
#define        ADR_WR_X_LOCAL_SCALE               (0x10<<2) 
#define        ADR_WR_M_LOCAL_SCALE               (0x11<<2) 
#define        ADR_GAMMA_LOCAL_SCALE              (0x12<<2) 
#define        ADR_BETA_LOCAL_SCALE               (0x13<<2) 
#define        ADR_B_LOCAL_SCALE                  (0x14<<2) 
#define        ADR_W_LOCAL_SCALE                  (0x15<<2) 
#define        ADR_LR_MODEL_SCALE                 (0x16<<2) 
#define        ADR_GAMMA_GLOBSCALE                (0x17<<2) 
#define        ADR_XGLOBAL_SCALE                  (0x18<<2) 
#define        ADR_MGLOBAL_SCALE                  (0x19<<2) 
#define        ADR_BETAGLOB_SCALE                 (0x1a<<2)   //                              ,
#define        ADR_WGLOBAL_SCALE                  (0x1b<<2)   //   [ 7:0]                     ,//选择某个编号的目标进行参数设罿
#define        ADR_QGLOB_SCALE                    (0x1c<<2)   //                              ,
#define        ADR_BGLOBAL_SCALE                  (0x1d<<2)   //   [DAT_WIDTH:0]                     ,
#define        ADR_GSTF_WEIGHT_SCALE              (0x1e<<2)   //   [DAT_WIDTH:0]                     ,
#define        ADR_HTOT_SCALE                     (0x1f<<2)   //   [DAT_WIDTH:0]                     ,

#define        ADR_ALGOUT_DAT0                    (0x20<<2)
#define        ADR_ALGOUT_DAT1                    (0x21<<2)
#define        ADR_ALGOUT_DAT2                    (0x22<<2)
#define        ADR_ALGOUT_DAT3                    (0x23<<2)
#define        ADR_ALGOUT_DAT4                    (0x24<<2)
#define        ADR_ALGOUT_DAT5                    (0x25<<2)
#define        ADR_ALGOUT_DAT6                    (0x26<<2)
#define        ADR_ALGOUT_DAT7                    (0x27<<2)
#define        ADR_ALGOUT_DAT8                    (0x28<<2)
#define        ADR_ALGOUT_DAT9                    (0x29<<2)
#define        ADR_ALGOUT_DAT10                   (0x2a<<2)
#define        ADR_ALGOUT_DAT11                   (0x2b<<2)
#define        ADR_ALGOUT_DAT12                   (0x2c<<2)
#define        ADR_ALGOUT_DAT13                   (0x2d<<2)
#define        ADR_ALGOUT_DAT14                   (0x2e<<2)
#define        ADR_ALGOUT_DAT15                   (0x2f<<2)
#define        ADR_ALGOUT_DAT16                   (0x30<<2)   
#define        ADR_TIM                            (0x31<<2)

#define        ADR_HW_STATUS                      (0x80<<2) 
  
#define        ADR_XORDER_PTRIM_RAM_L             (0x200<<2)
#define        ADR_XORDER_PTRIM_RAM_H             (0x600<<2)
 
#define        ADR_LR_MODEL_RAM_L                 (0x700<<2)
#define        ADR_LR_MODEL_RAM_H                 (0x1b00<<2)
 
#define        ADR_BETA_RAM_L                     (0x1c00<<2)
#define        ADR_BETA_RAM_H                     (0x1d00<<2)
 					       
#define        ADR_BG_RAM_L                       (0x1e00<<2)
#define        ADR_BG_RAM_H                       (0x1f00<<2)
 
#define        ADR_GST_RAM_L                      (0x2000<<2)
#define        ADR_GST_RAM_H                      (0x2100<<2)  


using namespace std;
 

int verbose = 0;
int            fdC2H, fdH2C;
int            fdReg;
char          *allocated = NULL;
uint8_t*      map=NULL;
float X_global_scale[100];
uint32_t ram_ptrim[492];
uint32_t ram_lrmodel[2544];

std::vector<uint32_t> ram_coeff;
std::vector<uint16_t> ram_coeff_b16;
uint32_t ram_bglobal[12];
uint32_t ram_gsfweight[12];
float X_local_scale[100];
float htot_gold[17*100];
#define DELAY_S(n) std::this_thread::sleep_for(std::chrono::seconds(n))
#define DELAY_MS(n) std::this_thread::sleep_for(std::chrono::milliseconds(n))
#define DELAY_US(n) std::this_thread::sleep_for(std::chrono::microseconds(n))

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

template<typename ... Args>
std::string strFormat(const std::string& format, Args ... args){
    size_t size = 1 + snprintf(nullptr, 0, format.c_str(), args ...);  // Extra space for \0
    char bytes[size];
    snprintf(bytes, size, format.c_str(), args ...);
    return std::string(bytes);
}

float hexToFloat_ptr(uint32_t hex) {
    return *reinterpret_cast<float*>(&hex);
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




void RegWr(uint32_t u32Adr,uint32_t u32Dat)
{
	// std::lock_guard<std::mutex> guard(mutexWreg);
    // std::string strLog;
    DELAY_US(10);
    *((uint32_t *)(map+ u32Adr)) = u32Dat;

    //  strLog=strFormat("%s: W[0x%x]=0x%x",__func__,u32Adr,u32Dat);
    //  LOG(INFO)<<strLog;
}
uint32_t RegRd(uint32_t u32Adr)
{
	// std::lock_guard<std::mutex> guard(mutexRreg);
    DELAY_US(10);
    return *((uint32_t *) (map+ u32Adr));
}

void File2DDR(string& strFile,uint32_t ddr_sta_adr)
{
	    uint64_t tot_size;
	    // printf("strFile =%s\n",strFile.c_str());
	    file_base file(strFile,FILE_DIR::FIN,FILE_TYPE::BINARY);		
 

		tot_size = file.readAll(allocated);
		// printf("%s'tot_size=0x%x\n",strFile.c_str(),tot_size);
		// printf("read after:%s'tot_size=0x%x\n",strFile.c_str(),tot_size);
        int s32Ret  =  write_from_buffer(FPGA_H2C_NODE, fdH2C, (char *)allocated, tot_size, ddr_sta_adr);
        // printf("%s load to ddr,0x%x done\n",strFile.c_str(),ddr_sta_adr);
}
void LoadGoldDat()
{
		std::string strFile;
	    uint64_t tot_size;
	    strFile  = "/home/hzhy/cpp_work/dat/h_total_float.dat";
	    file_base file(strFile,FILE_DIR::FIN,FILE_TYPE::BINARY);		
		tot_size = file.readAll((char*)htot_gold);
		// printf("LoadGoldDat:%s'tot_size=0x%x\n",strFile.c_str(),tot_size);	

} 
uint32_t floatToHex(float f) {
    uint32_t hexValue;
    memcpy(&hexValue, &f, sizeof(float)); // 将浮点数的内存布局复制到整数
    return hexValue;
}
void NetRegInit()
{
//or_wr_X_local_scale            <= 32'h44196454 ;//                  )//input       [31:0]      //=  32'h44196454;
    RegWr(ADR_WR_M_LOCAL_SCALE ,0x454fcb4d);//or_wr_M_local_scale            <= 32'h454fcb4d ;//                  )//input       [31:0]      //=  32'h454fcb4d;
    RegWr(ADR_GAMMA_LOCAL_SCALE,0x471f987c);//or_gmma_localscale             <= 32'h471f987c ;//                  )//input       [31:0]      //=  32'h471f987c;//40856.48631405182;
    RegWr(ADR_BETA_LOCAL_SCALE ,0x473389d2);//or_beta_local_scale            <= 32'h473389d2 ;//                  )//input       [31:0]      // = 32'h473389d2;//45961.8203125;
    RegWr(ADR_B_LOCAL_SCALE    ,0x47c8e2b8);//or_b_local_scale               <= 32'h47c8e2b8 ;//                  )//input       [31:0]      // = 32'h47c8e2b8;//102853.4375;
    RegWr(ADR_W_LOCAL_SCALE    ,0x483040e4);//or_w_local_scale               <= 32'h483040e4 ;//                  )//input       [31:0]      // = 32'h483040e4;//180483.5625
    RegWr(ADR_LR_MODEL_SCALE   ,0x477ffe00);//or_lr_model_scale              <= 32'h477ffe00 ;//                  )//input       [31:0]      // = 32'h477ffe00;//65534.0
    RegWr(ADR_GAMMA_GLOBSCALE  ,0x478a28a4);//or_Gamma_globscale             <= 32'h478a28a4 ;//                  )//input      [ 31:0]                              //                                                                       
    //or_Xglobal_scale               <= 32'h432239d4 ;//                  )//input      [ 31:0]                              //                                                                     
    RegWr(ADR_MGLOBAL_SCALE    ,0x444cf3b8);//or_Mglobal_scale               <= 32'h444cf3b8 ;//                  )//input      [ 31:0]                              //                                                                     
    RegWr(ADR_BETAGLOB_SCALE   ,0x484bbac5);//or_betaglob_scale              <= 32'h484bbac5 ;//                  )//input      [ 31:0]                              //                                                                      
    RegWr(ADR_WGLOBAL_SCALE    ,0x489dacf6);//or_Wglobal_scale               <= 32'h489dacf6 ;//                  )//input      [ 31:0]                              //                                                                     
    RegWr(ADR_QGLOB_SCALE      ,0x48bfc446);//or_Qglob_scale                 <= 32'h48bfc446 ;//                  )//input      [ 31:0]                              //                                                                       
    RegWr(ADR_BGLOBAL_SCALE    ,0x484f719e);//or_bglobal_scale               <= 32'h484f719e ;//                  )//input      [ 31:0]                              //                                                                         
    RegWr(ADR_GSTF_WEIGHT_SCALE,0x47d553aa);//or_gstf_weight_scale           <= 32'h47d553aa ;// 
}
 
 

void load_datfile2coeff_b16(const std::string& strFilename,uint32_t reg_adr,uint16_t reg_len)
{
	std::string strFile;
	std::vector<uint16_t> reg_arr;
	uint64_t tot_size;
	strFile  = "/home/hzhy/cpp_work/dat/"+strFilename;
 	printf("strFile =%s\n",strFile.c_str());
	file_base file(strFile,FILE_DIR::FIN,FILE_TYPE::BINARY);		
	tot_size = file.readAll(ram_coeff_b16);
	for(int i=0;i<reg_len;i++)
	{
		RegWr(reg_adr+i*4,ram_coeff_b16[i]);
	//	std::cout<<"ram_coeff["<<std::hex<<reg_adr+i*4<<"]="<<ram_coeff_b16[i]<<std::endl;
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

void Start()
{

    RegWr(ADR_ALG_START,0x0);
	DELAY_MS(10);
	RegWr(ADR_ALG_START,0x1);
}
void Reset()
{

    // RegWr(ADR_LOGIC_RST,0x1);
	DELAY_MS(10);
	RegWr(ADR_LOGIC_RST,0x0);
}

void LoadScale()
{
		std::string strFile;
	    uint64_t tot_size;
	    strFile  = "/home/hzhy/cpp_work/dat/X_global_scale_all100.dat";
 	    printf("strFile =%s\n",strFile.c_str());
	    file_base file(strFile,FILE_DIR::FIN,FILE_TYPE::BINARY);		
		tot_size = file.readAll((char*)X_global_scale);
 
		// for(int i=0;i<10;i++)
		// {
		// 	printf("X_global_scale[%d]=%f\n",i,X_global_scale[i]);
		// }
		strFile  = "/home/hzhy/cpp_work/dat/X_local_scale_all100.dat";
 	    printf("strFile =%s\n",strFile.c_str());
	    file_base file1(strFile,FILE_DIR::FIN,FILE_TYPE::BINARY);		
		tot_size = file1.readAll((char*)X_local_scale);
 
		// for(int i=0;i<10;i++)
		// {
		// 	printf("X_local_scale[%d]=%f\n",i,X_local_scale[i]);
		// }		
}
// #define        ADR_WR_X_LOCAL_SCALE               (0x10<<2) 
// #define        ADR_WR_M_LOCAL_SCALE               (0x11<<2) 
// #define        ADR_GAMMA_LOCAL_SCALE              (0x12<<2) 
// #define        ADR_BETA_LOCAL_SCALE               (0x13<<2) 
// #define        ADR_B_LOCAL_SCALE                  (0x14<<2) 
// #define        ADR_W_LOCAL_SCALE                  (0x15<<2) 
// #define        ADR_LR_MODEL_SCALE                 (0x16<<2) 
// #define        ADR_GAMMA_GLOBSCALE                (0x17<<2) 
// #define        ADR_XGLOBAL_SCALE                  (0x18<<2) 
// #define        ADR_MGLOBAL_SCALE                  (0x19<<2) 
// #define        ADR_BETAGLOB_SCALE                 (0x1a<<2)   //                              ,
// #define        ADR_WGLOBAL_SCALE                  (0x1b<<2)   //   [ 7:0]                     ,//选择某个编号的目标进行参数设罿
// #define        ADR_QGLOB_SCALE                    (0x1c<<2)   //                              ,
// #define        ADR_BGLOBAL_SCALE                  (0x1d<<2)   //   [DAT_WIDTH:0]                     ,
// #define        ADR_GSTF_WEIGHT_SCALE              (0x1e<<2)   //   [DAT_WIDTH:0]                     ,
 
int main()
{
    std::cout<<"Hello,World!"<<__DATE__<<","<<__TIME__<<std::endl;
 
    uint64_t       newTime, oldTime, loopNum = 0;
    int                s32Ret = 0;


    uint8_t        temp;
    bool           flag;
	ram_coeff.reserve(4096);



	fdReg =  open("/dev/xdma0_user", O_RDWR|O_SYNC);
    map =(uint8_t*)mmap(NULL, 131072, PROT_READ | PROT_WRITE, MAP_SHARED, fdReg,0x0);
    if (map == (void *)-1) {
            printf("Register Memory  mapped failed: %s.\n",  strerror(errno));
            close(fdReg);
			return -1;
    }
 
	int32_t dat = RegRd(0xc000);
	printf("FPGA VERSION =0x%x\n",dat);

	// close(fdReg);
	// return 0;

    /*将数据从主机写入到FPGA*/
    fdH2C = open(FPGA_H2C_NODE, O_RDWR);
    if(fdC2H < 0)
    {
        printf("Failed to open node %s!\n", FPGA_H2C_NODE);
        return -1;
    }
    else
    {        printf("Open node %s success\n", FPGA_H2C_NODE);
    }
  	posix_memalign((void **)&allocated, 1048576 /*alignment */ ,  HZHY_PS_BUF_LEN);
	printf("allocated'a address=%p\n",allocated);
	load_coeff();
	Reset();
	LoadGoldDat();//

	std::string strFile;
	// strFile  = "/home/hzhy/cpp_work/dat/XLocal.dat";
	// File2DDR(strFile,XLOCAL_ADR);

	strFile  = "/home/hzhy/cpp_work/dat/local_coef.dat";
	File2DDR(strFile,LOCAL_COEFF_ADR);

	strFile  = "/home/hzhy/cpp_work/dat/b_local.dat";
	File2DDR(strFile,BLOCAL_COEFF_ADR);
 
    strFile  = "/home/hzhy/cpp_work/dat/global_coef.dat";
	File2DDR(strFile,GLOBAL_COEFF_ADR);

	strFile  ="/home/hzhy/cpp_work/dat/Q_global.dat";
	File2DDR(strFile,QGLOB_COEFF_ADR);

	
	// strFile  = "/home/hzhy/cpp_work/dat/x_global.dat";
	// File2DDR(strFile,XGLOB_ADR);

	NetRegInit();


	dat = RegRd(ADR_ALG_START);
	printf("ADR_ALG_START=%x\n",dat);

	dat = RegRd(ADR_HW_STATUS);
	printf("ADR_HW_STATUS=%x\n",dat);

	

 	string str0 =  "XGlobal";
    string str1 =  ".dat";
	string strTp ;
	 LoadScale();
	 bool bErr=false;
	 int times =0; 
	std::vector<float> s_all;
	for(int loop=0;loop<100;loop++)
	{
	    printf("-------------------------------------\nloop=%d\n",loop);
		bErr =false;
		RegWr(ADR_WR_X_LOCAL_SCALE ,floatToHex(X_local_scale[loop]));//scale  update 
		RegWr(ADR_XGLOBAL_SCALE ,floatToHex(X_global_scale[loop]));
		dat = RegRd(ADR_WR_X_LOCAL_SCALE);
		float dat_fp =hexToFloat_ptr(dat);
		// printf("R[ADR_WR_X_LOCAL_SCALE]=%f\n",dat_fp);

		// printf("X_local_scale[%d]=%x\n",loop,floatToHex(X_local_scale[loop]));
		// printf("X_global_scale[%d]=%x\n",loop,floatToHex(X_global_scale[loop]));
		//load global data and local data
		str0 =  "XGlobal";
		str1 =  ".dat";
		strTp = "/home/hzhy/cpp_work/all_datfile/"+str0+strFormat("%d",loop)+str1;
		File2DDR(strTp,XGLOB_ADR );
		str0 =  "XLocal";
		// str0 =  "Xorder";
		strTp = "/home/hzhy/cpp_work/all_datfile/"+str0+strFormat("%d",loop)+str1;
		File2DDR(strTp,XLOCAL_ADR);
		Start();
	    DELAY_MS(100);
		
		float s_data = 0.0;
		float s =0.0;

		for(int i=0;i<17;i++)
		{
			dat = RegRd((0x20+i)<<2);
			dat_fp =hexToFloat_ptr(dat);
			
			s_data += sigmoid(dat_fp);

			// if(abs(dat_fp - htot_gold[times])>0.001)
			// {
			// 	bErr =true;
			// 	printf("R[%d]=%f,%f(gold)\n",i,dat_fp,htot_gold[times]);
			// }
	 
			times++;
		}
		s = s_data/17.0f;
		s_all.push_back(s);

		// printf("loop=%d,s_all=%f\n",loop,s_all);

		dat = RegRd(ADR_TIM);
		printf("R[ADR_TIM]=%f us\n",dat*4.1666/1000.0);
		 if(bErr==true)
			{
			    printf("Cmp ERROR!loop=%d\n",loop);
				// break;
			}
		// if(bErr==true)
		// 	break;
	}

	int n_positive = 50;//61
	int n_negative = 50;//1096
	int Ns= n_positive + n_negative;
	std::vector<int> label_test(Ns, 1);
	std::fill(label_test.begin() + n_positive, label_test.end(), 0);

	// 计算预测结果
	std::vector<int> y_predicted_final(Ns, 0);
	for (int n = 0; n < Ns; ++n) {
		y_predicted_final[n] = (s_all[n] >= 0.5);
	}

	// 计算准确率、TPR、FPR、BA
	int n_tp = 0;
	int n_fp = 0;
	for (int n = 0; n < Ns; ++n) {
		if (y_predicted_final[n] == 1 && label_test[n] == 1) n_tp++;
		if (y_predicted_final[n] == 1 && label_test[n] == 0) n_fp++;
	}

	float tpr = static_cast<float>(n_tp) / n_positive;
	float fpr = static_cast<float>(n_fp) / n_negative;
	float ba = (tpr + (1 - fpr)) / 2;
	float acc = static_cast<float>(n_tp + (n_negative - n_fp)) / Ns;

	// 计算AUC
	std::vector<float> fpr_1, tpr_1;
	for (float threshold = 0.0; threshold <= 1.0; threshold += 0.01) {
		int tp = 0, fp = 0;
		for (int n = 0; n < Ns; ++n) {
			if (s_all[n] >= threshold && label_test[n] == 1) tp++;
			if (s_all[n] >= threshold && label_test[n] == 0) fp++;
		}
		fpr_1.push_back(static_cast<float>(fp) / n_negative);
		tpr_1.push_back(static_cast<float>(tp) / n_positive);
	}
	float auc = 0.0;
	for (size_t i = 1; i < fpr_1.size(); ++i) {
		auc += (fpr_1[i] - fpr_1[i - 1]) * (tpr_1[i] + tpr_1[i - 1]) / 2;
	}
	auc = std::abs(auc);
	printf("BA=%f,ACC=%f,TPR=%f,FPR=%f,AUC=%f\n",ba,acc,tpr,fpr,auc);

	if(bErr==false)
		printf("-------------------------\nCmp OK!\n");
	else 
		printf("-------------------------\nCmp ERROR!\n");
	free(allocated);
	close(fdH2C);
}
