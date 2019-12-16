import math

class RNS:
    def __init__(self, bit_length):
        self.moduli = self.first_primes(bit_length)

    def transform(self, input):
        return [num % mod for num, mod in zip(input, self.moduli)]

    def is_prime(self, possible_prime, prime_list):
        until = math.sqrt(possible_prime)
        for prime in prime_list:
            if prime > until:
                break
            if possible_prime % prime == 0:
                return False
        return True

    def next_prime(self, prime_list):
        possible_prime = prime_list[-1]
        while True:
            possible_prime += 2
            if self.is_prime(possible_prime, prime_list):
                return possible_prime

    def first_primes(self, min_bit_length):
        prime_list = [2, 3]
        bit_length = math.log(6)
        while bit_length > min_bit_length:
            next_prime = self.next_prime(prime_list)
            prime_list.append(next_prime)
            bit_length += math.log(next_prime)
        return prime_list
