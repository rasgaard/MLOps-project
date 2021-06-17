from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.get(name="MLOps_day7",
               subscription_id='5a60d53a-0d60-4174-a594-5c65a8fc1cbe',
               resource_group='MLOps')

env = Environment("ws_MLOps_env")

packages = CondaDependencies.create(conda_packages =['pip'],
                                    pip_packages = ['azureml-defaults', 'transformers', 'torch','torchvision','sklearn','wandb','datasets', 'joblib'])

env.python.conda_dependencies = packages

script_config =ScriptRunConfig(source_directory = '.',
                                script ='src/models/train_model.py',
                                environment = env)

experiment = Experiment(workspace=ws, name = 'Training-160621')
run = experiment.submit(config =script_config)
run.wait_for_completion()