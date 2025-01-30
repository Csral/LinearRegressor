from sys import exit
from matrix import linalg, Matrix, zeros
import random
from d import feature, label

import pandas as pd

'''

Linear regressor formula:

(A'A)x = (A'B)

Here, A = [ x1 1 ; x2 1 ; x3 1 ; ... xn 1; ]
x = [ m ; c]
B = [ y1 y2 y3 ... yn ]

AX = B

A'A (X) = A'B

'''

class LinearRegressor:

    def __init__(self):
        self.A = None #* Place holder for X (feature)
        self.B = None #* Place holder for y (label)
        self.M = None #* Place holder for slopes.

    def _generic_length(self,data):
        # Returns the length of the first list in the dictionary values.
        return len(next(iter(data.values()))) if data else 0

    def _validate_data(self, data: dict) -> None:

        if not data.keys():
            raise TypeError("Empty data passed!")
        
        if not all(isinstance(x, (list)) for x in data.values()):
            raise TypeError("Invalid data provided. Expects values to be list of data points!")
        
        if not all( isinstance(y, (float,int)) for x in data.values() for y in x ):
            raise TypeError("Invalid data points. Expects integers!")

        generic_length = self._generic_length(data)

        for x in data.values():
            if len(x) != generic_length:
                raise ValueError("All features should have same amount of data points!\nPad them per need.")
            
        return data

    def _set_data_points(self, data: list, pointer: int) -> None:

        #* Accepts a dict of data points
        #* And a pointer (which parameter is this)
        #* 3 parameters means pointer goes 0,1,2.
        #* Sets pointer column with data

        if pointer > self.A.size()[1]: #* If pointer exceeds maximum columns
            raise ValueError("Parameter error: Pointer exceeded maximum columns!")

        self.A.set_column(pointer, data);

    def train(self, train_data: dict, y_true: dict) -> None:

        if not isinstance(train_data, dict):
            raise TypeError("Error: Invalid training data, expected a dict!")
        
        self._validate_data(train_data);
        
        self.M = zeros(
            len(train_data.keys())+1,
            1
        ) #* generate proper matrix to hold all slopes

        self.A = zeros( self._generic_length(train_data) , len(train_data)+1 )
        
        last_column = []

        for i in range(0, self._generic_length(train_data)):
            last_column.append(1) #* Prepare last column with 1's

        self.A.set_column(len(train_data), last_column) #* Set/update last column
        iter_pointer = 0

        for data_points in train_data.values():
            self._set_data_points(data_points, iter_pointer)
            iter_pointer += 1

        self._validate_data(y_true)

        if self._generic_length(y_true) != self._generic_length(train_data):
            raise ValueError("Count of data points in label and features should be same")

        #* Currently only supports 1 label

        self.B = zeros(self._generic_length(y_true), 1)

        for label_point in y_true.values():
            self.B.set_column(0, label_point)

        #* Establishing of marices is done.
        #* Now we solve for M using (A'A)*M = A'B or doing Gaussian elimination on [ A'A | A'B ]

        AT = linalg.transpose(self.A)
        ATA = linalg.multiply(AT,self.A) #* A'A
        ATB = linalg.multiply(AT, self.B) #* A'B

        self.M = linalg.solver(ATA,ATB)

    def predict(self, data: list) -> float:

        y = 0

        if not data:
            raise ValueError("Empty data provided for predict()")
        
        if len(data) != (self.M.size()[0]-1):
            raise ValueError("Invalid data: Number of elements given in data passed not equal to trained data")
        
        for i in range(0,len(data)):
            y += self.M.get(i,0)*data[i] #* m*x
        
        y += self.M.get(len(data),0) #* +c

        return y

l = LinearRegressor()

l.train(
    {
        1: [ 1,2,3,4,5,9,908,9808,898887,755,675 ],
        2: [ 2,3,5,6,10,6,9,98,546,87,34 ],
        3: [ 2,3,5,6,10,6,9,98,546,87,34 ],
        4: [ 2,3,5,6,10,6,9,98,546,87,34 ],
        5: [ 2,3,5,6,10,6,9,98,546,87,34 ]
    },
    {
        1: [ 1,2,3,4,6,8765,456,76,45,897,54 ]
    }
)

m = l.predict([1,2])

print("\n\n")
print(m)

'''

if __name__ == "__main__":
    print("Error: Cannot run this script!")
    exit(1)'''