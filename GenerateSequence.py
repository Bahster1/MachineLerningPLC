import csv
import random
from pandas import *

def generateData(points:int, sequence:list = None):
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["Time", "Data"]

        writer.writerow(field)

        if sequence is None:
            sequence = list()
            sequence_length = 100

            for i in range(sequence_length):
                sequence.append(random.randint(0, 12000))

        for i in range(points):
            writer.writerow([i, sequence[i % len(sequence)]])



def generateData2(points, sequenceName):
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ['Time', 'Data']
        data = read_csv("DATALOG.txt", delimiter=',')
        sequence = data[sequenceName].tolist()

        writer.writerow(field)

        for i in range(points):
            writer.writerow([i, sequence[i % len(sequence)]])
