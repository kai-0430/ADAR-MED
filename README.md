<div style="font-size: 450%;">
  <h1 align="center">👨‍⚕️ ADAR-MED 👩‍⚕️</h1>
</div>

<h3 align="center">
    An AI-Driven Assistant for Rapid Medical Diagnosis
</h3>
<p align="center">
    Links: <a href="https://www.hackster.io/contests/amd2023/hardware_applications/16954">Proposal</a> | <a href="https://www.hackster.io/519710/adar-med-ai-driven-assistant-for-rapid-medical-diagnosis-8f6e0c#toc-web-ui-4">Report</a> | <a href="https://github.com/kai-0430/ADAR-MED">GitHub</a><br />
</p>
<p align="center">
  <picture> <img alt="MED-ALPACA" src="https://github.com/user-attachments/assets/16ce174a-0217-411c-ae59-a4cdf3dd39fa" width=55%>
</p>
      
## Installation

1. (Recommended) Create a conda environment with Python version >= 3.11.
    ```
    conda create -n your_env python=3.11
    conda activate your_env
    ```
2. Install vllm. For installing vllm with ROCm on AMD accelerate cloud, you can check this [website](https://hackmd.io/@unj0M9DkQhqZGOyd71BT5g/HkFNSQEHR). If you are using other platforms, please visit the [vllm documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html).
3. Clone this repository
    ```
    git clone https://github.com/kai-0430/ADAR-MED.git
    cd ADAR-MED
    ```
4. Install ADAR-MED that's built on FastChat v0.2.36.
    ```
    pip3 install --upgrade pip
    pip3 install -e ".[model_worker,webui]"
    ```
## Model
You can download the MedAlpaca-7b model from [Huggingface](https://huggingface.co/medalpaca/medalpaca-7b).

## How to run ADAR-MED?
1. Open three terminals, activate the conda environment, and navigate to the ADAR-MED directory.
2. In terminal 1, launch the controller. On the AMD accelerate cloud, we use the host ip `127.0.0.1`.
    ```
    python3 -m fastchat.serve.controller --host 127.0.0.1
    ```
3. In terminal 2, launch the vLLM engine. This will take a while since we will load a model in this step.
    ```
    python3 -m fastchat.serve.vllm_worker --host 127.0.0.1 --model-path /path/to/med_alpaca --max-num-seqs 768
    ```
4. In terminal 3, launch the ADAR-MED web user interface.
    ```
    python3 -m fastchat.serve.med_chabot_web_server --host 127.0.0.1 --share
    ```
More argument details can be found in the [report](https://www.hackster.io/519710/adar-med-ai-driven-assistant-for-rapid-medical-diagnosis-8f6e0c#toc-web-ui-4). 

