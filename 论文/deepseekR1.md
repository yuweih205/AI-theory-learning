他在prompt里面加入了输入格式什么什么的 而不是直接训练出来的

训练过程

感觉写的不是很清楚，也可能是我没读懂。

* zero是纯GRPO

  *  为了节约计算成本(估计加上critic model也没啥太大的效果)，只有policy model，什么是policy，数学原理上我还不太明白，但是强化学习里面每走一步他会得到一个reward值，policy model负责告诉他下一步这个值reward的期望值是多少，G是group，意思是，每次输入，会生成一组候选，这组候选的平均值就是baseline，这样就可以知道哪个比这个base好，以此来更新策略，~~reward model也可以用v3来充当~~，这样可以学到各种回答的好坏直接**程度**
  *  其实最重要的就是reward model，采用了规则制定，两方面，一个是准确度，二是格式，不采取神经网络反馈模型是因为可能在RL里面会遇到reward hacking（只在RL里面有吗 啥时候会触发该bug），而且重新训练需要很多计算资源，让pipeline变得非常复杂

* R1是sft+GPRO 

  ```
  #Cold start 
  目的：可读性 CoT格式
  数据集：thousands of CoT data   design a readable pattrn：the output format as |special_token|<reasoning_process>|special_token|<summary>  ，with human priors 
  
  #RL：
  数据：
  还存在语言混用 
  引入了语言一致性reward combine the accuracy of reasoning tasks and the reward for language consistency by directly summing them to form the final reward
  
  #Rejection Sampling and Supervised Fine-Tuning
  培养非推理能力 综合领域知识
  在这时的checkpoint做了一些rejection sampling
  混入了很多别的领域数据集 引入了一些由绝对正确的rewad model生成的数据集 给V3评判 过滤了语言混用 长文本 代码块 拿到600k训练样本（这里看上去只有正例没有负例？） 
  还加入了200k训练集后推理无关的（选用了V3的数据集部分 用V3生成promt的具有CoT的回复）
  
  #RL： 这里用的什么样本 广泛的promt分布 models to capture human preferences in complex and nuanced scenarios 使用了很多不同的偏好对还有training promt 这个promt是什么 没有回答的吗 给出很多答案怎么评判呢
  除了增强推理效果。还需要对齐人类偏好 帮助性和无害性
  对于reasonging数据 沿用Zero的手写规则reward 
  对于一般数据 采取和v3差不多的偏好对齐（一组优劣）以训练promt的分布来训练  对于helpfulness 评估最后summary 对于harmlessness 打分全过程  就这些都是reward model？？
  总体来说 RL后的样本丢给sft
  ```
  
* 蒸馏也只做了sft，能力有所增强，据说加上强化学习会更强

  * 直接微调  由deepseek-R1拿到的800k样本 