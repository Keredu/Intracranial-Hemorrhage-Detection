from train import train_model
from config import get_config

# Get config from conf.yaml
conf = get_config('conf.yaml')

# Train and evaluate
model, hist = train_model(conf)

