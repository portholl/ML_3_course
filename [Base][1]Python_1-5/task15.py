from typing import List


def hello(s=None) -> str:
    if s: return("Hello, "+ s + "!")
    else: return("Hello!")



def int_to_roman(x) -> str:
    res = ""
    roman_map = {1000: 'M', 900: 'CM', 500: 'D', 400: 'CD', 100: 'C', 90: 'XC', 50: 'L', 40: 'XL', 10: 'X', 9: 'IX', 5: 'V', 4: 'IV', 1: 'I'}
    for value, symbol in roman_map.items():
        while x >= value:
            res += symbol
            x -= value
    return res


def longest_common_prefix(a) -> str:
    if not a : return ""
    ansv = a[0].strip()
    for i in a:
        i = i.strip()
        ans = ""
        for j in range(min(len(ansv), len(i))):
            if i[j] == ansv[j]:
                ans += ansv[j]
            else : break
        ansv = ans
    return ansv
                
def primes():
    i = 2
    while True:
        T = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                T = False
                break
        if T : 
            yield i
        i += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = -1):
        self.total_sum = total_sum
        self.balance_limit  = balance_limit
    def __call__(self, sum_spent: int):
        if self.total_sum >= sum_spent:
            print(f"You spent {sum_spent} dollars.")
            self.total_sum  -= sum_spent
        else: 
            print("Not enough money to spend sum_spent dollars.")
            raise ValueError
    def put(self, sum_put: int):
        print("You put sum_put dollars.")
        self.total_sum  += sum_put
    @property 
    def balance(self):
        if self.balance_limit < 0:
            return self.total_sum
        elif self.balance_limit > 0: 
            self.balance_limit -= 1
            return self.total_sum
        else:
            print("Balance check limits exceeded.")
            raise ValueError
    def __str__(self):
        return("To learn the balance call balance.")
    def __add__(self, a):
        return BankCard(self.total_sum + a.total_sum, balance_limit = max(self.balance_limit, a.balance_limit))

            

        

