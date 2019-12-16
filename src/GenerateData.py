from Crypto.Util import number
from LMGS import DataPair

class GenerateData:
    @staticmethod
    def generate_data(sample_size, bit_length):
        data = []
        for _ in range(sample_size):
            p = number.getPrime(bit_length // 2)
            q = number.getPrime(bit_length // 2)
            data.append(DataPair(p * q, min(p, q)))
        return data


# if __name__ == "__main__":
#     bit_length = 256
#     sample_size = 1000
#     train_data = generate_data(sample_size, bit_length)
#     priu