# This class runs multiple instances of the Godunov scheme, and plots the results of accuracy and speed

import sys

def main(terminal_argument):
    cells = [100*2**i for i in range(7)]
    

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")