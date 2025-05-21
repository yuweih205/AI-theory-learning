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





举个例子：GQA的实现

