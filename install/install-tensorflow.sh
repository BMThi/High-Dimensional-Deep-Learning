nvidia-smi 
# check that cuda version is 12.2 call the teacher if not.

yes | conda create -n tensorflow python=3.10
source activate tensorflow
pip install tensorflow[and-cuda]

# pour tester
python
import tensorflow as tf
tf.config.list_physical_devices('GPU')




