from src.generate_data import generate_data
from src.rns import RNS

bit_length = 256
data = generate_data(1000, bit_length)
rns = RNS(bit_length)
data = rns.transform()
