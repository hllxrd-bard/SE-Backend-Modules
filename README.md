# Environment Setup
- Python 3.10
- CUDA 12.6
## Setup PPDocLayout-v3
Running this:
```
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddleocr
```
## Setup other libraries
Running this:
```
pip install opencv-python pymupdf 
```
## Setup ssh-key for Summary LLM (Once for each Docker container)
Running this:
```
ssh-keygen
```
then just **Enter** x π until it finishes generating SSH key
then running this:
```
/usr/lib/autossh/autossh -M 0 -N -o ServerAliveInterval=60 -o ServerAliveCountMax=0 -L 0.0.0.0:5001:AsusL40:27304 aiclub@slurm.uit.edu.vn
```
it will require password, paste your password, then it wont print anything, dont stop, create new terminal, exec to tthe **same docker container** and run the code normally


