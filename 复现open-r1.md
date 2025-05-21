```
Your repo now lives at:
  https://huggingface.co/Aia2024/open-r1-grpo

You can clone it locally with the command below, and commit/push as usual.

  git clone https://huggingface.co/Aia2024/open-r1-grpo

(openr1) boss@th03-bluorion-w2:~/huangyuwei/open-r1$ CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 7     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml

```

