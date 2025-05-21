核心ray实现是指和ray框架集成相关的底层封装代码，比如如何用ray启动acttor，learner、服务进程等，

训练器trainer，封装了各种rl算法部分，指导模型更新

工作器worker，实际模型，实现actor的





用ray启动之后 ray job server接收到任务 创建一个job worker进程(driver 进程)，根绝runtime_env.yaml配置工作环境，执行verl.trainer.main.pop主函数，raytrainer读取训练配置，他就开始分配任务等
