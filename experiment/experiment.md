Here we explains how to reproduce throughput data of ADAR-MED server.
1. In `fastchat/serve/med_chabot_web_server.py`, send multiple concurrent requests.

   Original:

  ![image](https://github.com/user-attachments/assets/23d7fc9d-856f-4eef-a532-a87526bef67c)
  Revised:

  ![image](https://github.com/user-attachments/assets/9f7cf19a-11d4-4b35-8645-2f45897cd0f0)
2. In `vllm/engine/metrics.py` of your vllm package, record and output the `prompt_throughput`, `generation_throughput`, and `stats.num_running_sys`.

  ![image](https://github.com/user-attachments/assets/84329ec5-b0d1-4576-8641-6a9623c7bd49)
