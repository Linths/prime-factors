import math


def is_prime(possible_prime, prime_list):
    until = math.sqrt(possible_prime)
    for prime in prime_list:
        if prime > until:
            break
        if possible_prime % prime == 0:
            return False
    return True


def next_prime(prime_list):
    possible_prime = prime_list[-1]
    while True:
        possible_prime += 2
        if is_prime(possible_prime, prime_list):
            return possible_prime


def first_primes(min_bit_length):
    prime_list = [2, 3]
    bit_length = math.log(6)
    while bit_length > min_bit_length:
        next_prime_number = next_prime(prime_list)
        prime_list.append(next_prime_number)
        bit_length += math.log(next_prime_number)
    return prime_list


class RNS:
    def __init__(self, bit_length):
        self.moduli = first_primes(bit_length)

    def transform(self, input_numbers):
        return [num % mod for num, mod in zip(input_numbers, self.moduli)]
