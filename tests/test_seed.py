from torch import set_default_dtype
from src.data.make_dataset import seed_everything, read_data

def test_seed():
    x = read_data()
    seed_everything(0)
    y = read_data()
    assert x != y
    seed_everything(0)
    x = read_data()
    assert x == y



