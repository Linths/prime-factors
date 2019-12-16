from src.GenerateData import GenerateData
from src.RNS import RNS

bit_length = 256
data = GenerateData().generate_data(1000, bit_length)
rns = RNS(bit_length)
data = rns.transform()