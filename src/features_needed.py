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

def report_features(report_times):
    prime_list = [2, 3]
    bit_length = math.log(6, 2)
    for rt in report_times:
        while True:
            prime = next_prime(prime_list)
            prime_list.append(prime)
            bit_length += math.log(prime, 2)
            if bit_length > rt:
                num_params = 3 + 4 * len(prime_list)
                total_params = num_params * len(prime_list)
                print("Bit Length: " + str(rt) + " - # Parameters per Model: " + str(num_params) + " - " +
                        "# Total Parameters: " + str(total_params) + " - Largest Prime: " + str(prime))
                break
    print(prime_list)

report_times = [2 ** i for i in range(5, 14)]
between_times =  [(x + y) // 2 for x, y in zip(report_times, report_times[1:])]
report_times += between_times
report_times.sort()
report_features(report_times)