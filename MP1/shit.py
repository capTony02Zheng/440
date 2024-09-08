import numpy as np
def main(args):
    print(args)
    print(1+1)
    x = np.arange(9).reshape(3,3)
    print(x)
    print(np.where(x==3)[1])
    return 0
main("Shit my pant")