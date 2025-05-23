* long-context task

指的是输入篇幅很长、上下文范围跨度大(强调有用信息之间相隔很远)

* instruction-based reasoning

  比如，让模型具有Cot的能力

* NSA在推理、前向传播、后向传播都有很大的加速？？



* softmax内在的稀疏性，由于他会让大的变大，小的变小，很多都会趋近于0



这几篇论文可以再看一下

* sampling, clustering or hashing-based selection methods
  - 不太懂 这个怎么对token表示做聚类 只选每个类的代表token 避免冗余计算
* blockwise KV-cache selection methods
* KV-cache eviction methods



* Training-aware algorithm design
  * 这个可以回答我之前的疑问 为什么要从训练开始调整推理效率   在设计模型架构或优化算法时，不只是考虑推理效率，而是从训练阶段就考虑进去 说明存在这样的实现 加速训练和推理并且不影响模型精度
  
* architectural bias指什么
  
  因为模型的结构是在 Full Attention 下预训练出来的，模型的表示方式、注意力模式、甚至参数分布，都已经适应了 Full Attention 架构的行为习惯。

* MQA所有head共享同一组K/V    GQA是每组

* ClusterKV使用Kmeans聚类对KV进行分组，只保留代表性的聚类中心进行计算，
* retrieval heads 在注意力机制中负责检索定位相关信息的子结构

* kv聚类和magicpig在token的选择中无法微分，说明稀疏模式是静态的，而不是训练中优化出来的   逻辑链是什么样的？
  * attention的计算公式

* Hashattention 由于token粒度的选择策略，每个query访问的位置不连续 导致内存访问开销大 无法和falshattention结合使用

  * FA计算原理：

* 其实就分为三类：

  * 压缩     选择什么方式压缩    mlp 估计是mlp降维压缩  而且是直接压缩成一个数字？   如果复现的话 这个怎么评估压缩策略 或者学习到的数据没有问题  mlp放几层线性层  如果有激活函数 非线性层 就更复杂了  这样不是还是要遍历一遍吗 推理的时候呢    q不压缩  压缩kv     滑动步长可能有重叠   这个也是要确认的参数 （这些不确定的参数是根据硬件来优化的还是超参数也需要不断选择的？）

  * 选择性块稀疏 动态   fine-grained细粒度

    - 为什么说key value是成对的  还是不清楚

      gpt说计算query和key的相似度，就能检索到值，所以kv成对。可实际上注意力查询的时候一个query会遍历所有的value，得到一张注意力地图，放入softmax再和value打分，

    - 分块如何在gpu上加速运算还是需要细致谨慎的了解

    - 还得去了解一下flashattention的实现和原理 btw linear attention也要了解一下

    - 切分kv成选择块，打分重要性， overhed 开销。  为啥这个kvcache的情况 是一个累加的，在应用gqa和mqa的情况下 所以还得去读下kvcache的论文 

    - 这样保证组内的所有注意力头都选择相同的块计算 这样可以复用块  最后的块拼接起来 topn 按什么顺序拼接 而且这里的数据也是压缩的数据

    - 这样就是把压缩和选择性得一起 边压缩边打分 

  * 静态稀疏  

    *  选择当前token的k、v的之前的若干个token（没有之后的对吧？？），单独计算。

  最后通过门控机制来整合上面三个

这个块的大小怎么划定



##### kenerl design

对于每个内循环，加载所有头的query，在t位置的组，和他们的稀疏kv



 为啥在大模型的训练初期，第一层使用 MoE 可能导致训练不稳定



举个例子：GQA的实现



旋转位置编码不太理解，还有Vit中的旋转位置编码

即便q不参与梯度更新，kv也会参与？



在 LongBench(长上下文ben)，SQA单文档，MQA多文档，Synthetic（需要在大量无关文本中找到特定信息)。

还有很多方法，H2O, KV-cache eviction, query-aware selection, and exact top-𝑛 sparse selection.

为啥稀疏注意力baselines 不支持训练 应该是以前的不支持训练吧

* 评估的温度和top-p value是什么

 	温度低倾向于选择概率最高的词，高温度更加随机，倾向于引入更多随机性，top-p采样，模型将所有可能的下一个词按概率从高到低排序，选择累计概率达到p的最小集合，然后随机选一个

* 这个梯度更新是怎么做的 那部分稀疏的



* 这个避免灾难性遗忘还需要多多调试感觉 粒度





* Coalesced loads 这是 CUDA 中的一个**内存访问优化概念**，指同一个 warp（32 个线程）访问全局内存时，尽可能访问连续地址，从而合并为一条内存读指令。





感觉NSA带来的收益可能只是因为训练的时候就这样做了 其他方案做了一样有效果

* 注意力高分段成块状聚类分布，但看图，横竖都不一定啊，为啥时每行切，不是每列切  为啥每行是一个token 但是看图感觉更倾向于列状分块 

* 为啥启发性无参数的重要性打分策略容易导致召回率低  ？？ 
  * 启发式策略无法自适应复杂语境，容易漏掉真正有用但“看起来不重要”的 token 或 expert 那为啥还用
* 什么叫auxiliary loss-based selecton
  * 辅助损失驱动的token/block选择机制，不是用启发式打分(如范数)，额外训练目标来学习token/block的重要性分数
* 监督信号是什么
  * 目标分布要逼近的分布 类似标准答案
* attention sink是指在流式编码中扮演汇聚点(sink)角色的token 不随滑动窗口而被丢弃 Xiao et al 发现，出事token能吸引非常高的attention权重，尽管语义不重要，但能让模型对全局信息的分布接近原先的训练状态，从而稳定性能
* 前后窗口vs前面窗口
  * Encoder-style Attention 天生就是双向的：模型在做分类、QA、检索增强等任务时，能同时利用一个词前后的上下文信息。如果只看“前面”的 Token，主要用在文本生成/解码里；Longformer 的论文是针对长序列编码 (e.g. 文档分类、阅读理解)，所以用对称窗口。

* 感觉后面这些方法并不是不行 只是没有在训练的时候就应用进去
* HashAttention (Desai et al., 2024) formulates pivotal token identification as a recommendation problem by mapping queries and keys to Hamming space using learned functions. 这个能不能做一个专用于推荐的改进？？
* 梯度累积是什么 什么时候用 怎么优化训练时  训练时的kvcache
* 为啥不直接搞个多层attention  额mla就是





FLash attention和linear attention  chunk  page attention



算法实现：

1、压缩 q v分块 每块压缩成一个值  这个压缩方法有很多  但这样会导致信息碎片化或者说隔离 可以让block长度大于切割长度  论文里面说的是按mlp 这个这里的计算强度分析    这个mlp能不能只输入seq

2、 天然的空间连续性  这种连续性会因为别的xx而改变吗 那个特征图感觉不太靠谱？前面也有论文证明这个 

原始q乘以压缩的k 变成注意力  如果有交叠部分  做一个平滑加权计算  周围所有的block都算一遍？？？  选择前topn个计算全量注意力

 3、前n个结构化稀疏



还是不太清楚他是怎么实现梯度回传 压缩更新的

对dimension压缩的时候 能不能说明两件事情  一是通道维度附近本身就有相似性  二是可能不需要这么多维度就能达到一样的效果

