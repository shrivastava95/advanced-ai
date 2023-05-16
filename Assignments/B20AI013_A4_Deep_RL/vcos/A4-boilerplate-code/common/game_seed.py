roll = 'B20AI061'

seed = ""
for c in roll:
    if c.isdigit():
        seed += c
    else:
        seed += str(ord(c)-64)

print(seed)