H100

tflops   10<sup>12</sup> 浮点计算 模型训练 精度高支持训练 tensor core

1TOPS  10<sup>12</sup> 任意计算  整数  精度低 适合推理 

* 为啥会有这个区别 

![image-20250523155016366](/Users/admin/Library/Application Support/typora-user-images/image-20250523155016366.png)

3.35TB/s的带宽 10<sup>12</sup> byte/s  带宽指的是GPU每秒显存➡️alu/reg

nvlink 900GB/s    GPU ➡️ GPU 

PCIe Gen5 128 GB/s  GPU ➡️ CPU

这个最大散热设计功率 700W

* DPX指令 用来支持DP/匹配类计算的加速设计的专用硬件单元    动态选择 选择推理分支 动态选择token、attention mask、span prediction

* TMA指令 直接把tensor以tiled为单位 搬到shared memory

* 完整的GH100架构和SXM5主板封装的并不一样  是他的裁剪版
  * 8个GPC、66个TPC、132个SM（层层包含，一个GPC可能有8个TPC，16个SM，2048个fp 32 cuda core），还有16896个FP32 Core(14次方)（无损甚至更快执行低精度浮点计算，但低精度整数会慢，在计算前把它转化成浮点数)，528个Tensor Core，5个HBM堆栈，L2cache是50MB、L1cache 128-192（与share memory共用） shared memory+L1cache 总共256KB  
  * L0指令缓存是SM中不能直接控制 但可以间接影响的 几KB   一条指令一般是4bytes   每个clock cycle负责调度一个warp gpt推测是2-4KB 也就是500-1000条指令存

* gpu执行流 

  * warp scheduler每个周期调度上来一条指令 执行的时候 分配单元分配给寄存器文件 去load数据等等 然后送上计算单元 这个过程都是流水线式执行的  

  * gpt说明的流程：

  一个cluster里面  Warp Scheduler 发射 warp W 的指令 i（比如 add.f32 R1, R2, R3）
    ↓ Issue 分配器（dispatch unit）决定发去哪类执行单元（INT32/FP32/LD/ST/...）
    ↓ 寄存器文件准备读数据（R2、R3）
    ↓ Operand fetch & decode
    ↓ 运算在计算单元进行
    ↓ 写回寄存器（R1

* 一个SM里面有四个clsuter  共享一个L1指令缓存(16-32KB)    和  256KB  L1数据缓存/共享内存  两者动态平衡     

  * 还有个TMA  没有大小  一个加速器 16路pipeline支持 高效memory➡️ shared copy      

    global memory是GPU上的全局内存 也就是显存 所有SM共享     每个SM都有个TMA  

  * tile cache是什么

  
