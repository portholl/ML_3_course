def check(x: str, file: str):
    f = open(file,"w")
    x = x.split()
    x = sorted(x)
    d = {}
    for i in range(len(x)):
        x[i] = x[i].lower()
    for i in x:
        count = x.count(i)
        d[i] = count
    for i in d:
        print(i,d[i], file = f)
