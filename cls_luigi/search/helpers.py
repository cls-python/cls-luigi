def set_seed(seed: int = 42):
    import random
    import numpy as np
    # import torch

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
