nvidia-smi 
# check that cuda version is 12.2 call the teacher if not.

yes | conda create -n torch python=3.10
source activate torch
pip install torch torchvision torchaudio

# pour tester
python
import torch
torch.cuda.is_available() #must be true
