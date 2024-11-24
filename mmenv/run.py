from mmengine.config import Config
from mmengine.runner import Runner


cfg = Config.fromfile("./config/config.py")
cfg.launcher = "none"
cfg.work_dir = "checkpoints"

# resume training
cfg.resume = False
runner = Runner.from_cfg(cfg)

# start training
runner.train()