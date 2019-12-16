from Crypto.Util import number
from src.LMGS import DataPair

class GenerateData:
    def generate_data(self, sample_size, bit_length):
        data = []
        for _ in range(sample_size):
            p = number.getPrime(bit_length // 2)
            q = number.getPrime(bit_length // 2)
            data.append(DataPair(p * q, min(p, q)))
        return data
