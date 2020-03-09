import os
import sagemaker
from sagemaker.tensorflow import Tensorflow


sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
local_instance_type = "local"
remote_instance_type = "ml.t2.large"

source_dir = os.getcwd()
local_data_dir = os.path.join(os.getcwd(), "resources", "sample_input")
remote_data_dir = "aiops-2020/data-lake/twitter/state=processed/"

estimator = Tensorflow(
    entry_point="model_training/sentiment_training.py",
    source_dir=source_dir,
    role=role,
    framework_version="1.14.0",
    py_version="py3",
    hyperparameters={"num_epoch": 10},
    train_instance_count=1,
    train_instance_type=remote_instance_type
)

remote_inputs = {
    "train": "s3://" + remote_data_dir + "/train/",
    "validation": "s3://" + remote_data_dir + "/validation/",
    "eval": "s3://" + remote_data_dir + "/eval/",
}
estimator.fit(remote_inputs)