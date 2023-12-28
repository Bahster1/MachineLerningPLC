import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("DATALOG.txt", delimiter=",")

    for col in data.columns:
        plt.figure(num=col)
        plt.plot(data[col])
    
    plt.show()


if __name__ == "__main__":
    main()
