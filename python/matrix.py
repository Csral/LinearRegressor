from __future__ import annotations
from sys import exit
from random import randint

def zeros(r: int,c: int) -> Matrix:

    ans = [[0] * c for _ in range(r)]

    for i in range(0,r):
        for j in range(0,c):
            ans[i][j] = 0
    
    mans = Matrix(ans)
    return mans

def copy_matrix(matrix: Matrix) -> Matrix:
    #* Returns a temporaily new instance of given matrix
    
    ans = []
    temp_row = []

    r,c = matrix.size()

    for i in range(0,r):
        for j in range(0,c):
            temp_row.append(matrix.get(i,j))
        ans.append(temp_row.copy()) #* Seperate rows
        temp_row.clear() #* Clear row

    mans = Matrix(ans)
    return mans

def zeros_like(matrix: Matrix) -> Matrix:

    r,c = matrix.size()
    ans = []

    for i in range(0,r):
        ans.append([])

    for i in range(0,r):
        for j in range(0,c):

            ans[i][j] = 0
    
    mans = Matrix(ans)
    return mans

class LinearAlgebra:

    def __init__(self):
        self.matrix = Matrix([[0,0],[0,0]]); #* Place holder for matrix. Updates and used for internal calculation

    def elementary_multiply(self, matrix: Matrix, multiplier:int, row = -1, column = -1, no_update = False):
        #* Passing -1 would set all of that row/Columns

        r,c = matrix.size()
        row_wise = False
        column_wise = False

        if row == -1:
            row_wise = True
        
        if column == -1:
            column_wise = True

        if not (row_wise or column_wise):
            raise ValueError("Cannot multiply a single element of matrix!");

        if row_wise and column_wise:

            #* Multiply all elements of matrix:
            ans_val = []
            temp_row = []

            for i in range(0,r):
                for j in range(0,c):
                    val = matrix.get(i,j)
                    val *= multiplier

                    if not no_update:
                        #* Allowed to live update matrix:
                        matrix.set(val, i,j);
                    else:
                        #* Not allowed to update matrix, return a list.
                        temp_row.append(val)

                ans_val.append(temp_row.copy()) #* Move to new row
                temp_row.clear() #* Clear/empty row for new row
            
            if no_update:
                return ans_val

        elif row_wise and not column_wise: #* Multiply column.

            #* matlab eq: M[:,column]

            temp = []

            for i in range(0,r):
                val = matrix.get(i,column)
                val *= multiplier
                if not no_update:
                    matrix.set(val,i,column);
                else:
                    temp.append(val)
            
            if no_update:
                return temp

        elif not row_wise and column_wise:

            temp = []

            for i in range(0,c):
                val = matrix.get(row,i)
                val *= multiplier
                if not no_update:
                    matrix.set(val, row, i);
                else:
                    temp.append(val)
    
            if no_update:
                return temp

    def elementary_divide(self, matrix: Matrix, divisor:int, row = -1, column = -1, no_update = False):
        #* Passing -1 would set all of that row/Columns

        r,c = matrix.size()
        row_wise = False
        column_wise = False

        if row == -1:
            row_wise = True
        
        if column == -1:
            column_wise = True

        if not (row_wise or column_wise):
            raise ValueError("Cannot divide a single element of matrix!");

        if row_wise and column_wise:

            ans_val = []
            temp_row = []

            #* Divide all elements of matrix:
            for i in range(0,r):
                for j in range(0,c):
                    val = matrix.get(i,j)
                    val /= divisor

                    if not no_update:
                        #* Allowed to live update matrix:
                        matrix.set(val, i,j);
                    else:
                        #* Not allowed to update matrix, return a list.
                        temp_row.append(val)

                ans_val.append(temp_row.copy()) #* Move to new row
                temp_row.clear() #* Clear/empty row for new row
            
            if no_update:
                return ans_val

        elif row_wise and not column_wise: #* Divide column.

            #* matlab eq: M[:,column]

            temp = []

            for i in range(0,r):
                val = matrix.get(i,column)
                val /= divisor

                if not no_update:
                    matrix.set(val,i,column);
                else:
                    temp.append(val)
            
            if no_update:
                return temp

        elif not row_wise and column_wise:

            temp = []

            for i in range(0,c):
                val = matrix.get(row,i)
                val /= divisor

                if not no_update:
                    matrix.set(val,row,i);
                else:
                    temp.append(val)
            
            if no_update:
                return temp
    
    def swap_rows(self, src, dest) -> None:
        
        r1 = self.matrix.get_row(src)
        r2 = self.matrix.get_row(dest)

        self.matrix.set_row(src, r2)
        self.matrix.set_row(dest, r1)

    def rref(self, matrix: Matrix) -> Matrix:

        self.matrix = copy_matrix(matrix=matrix) #* Store current matrix as temp

        rows, columns = self.matrix.size();
        cindex = 0 #* Assume first column to have non-zero value.

        for row in range(rows):

            if cindex > columns:
                break

            #* Iterate through the rows
            rindex = row

            while self.matrix.get(rindex,cindex) == 0:
                rindex += 1
                if rindex == rows: #* No pivotal point here
                    rindex = row
                    cindex += 1
                    if cindex == columns: #* No more rows/columns left.
                        return self.matrix
            
            self.swap_rows(row,rindex)
            # Normalize the pivot row (make leading coefficient = 1)
            self.elementary_divide(self.matrix, self.matrix.get(row,cindex), row, -1) #* Divide current row with the pivotal point of current row.
            
            for i in range(rows):
                if i != row:
                    current_row = self.elementary_multiply(self.matrix, self.matrix.get(i,cindex), row, -1, no_update=True)
                    looper_row = self.matrix.get_row(i)

                    for x in range(0,columns):
                        self.matrix.set(
                            looper_row[x]-current_row[x],
                            i,
                            x
                        ); #* Subtract pivotal row * current leading element and place it back in current row.

            cindex += 1

        return self.matrix
    
    def multiply(self, a: Matrix, b: Matrix) -> Matrix:

        r1,c1 = a.size()
        r2,c2 = b.size()

        if c1 != r2:
            raise ValueError("Matrix product is not possible between the given matrices")
        
        ans = zeros(r1,c2) #* Create an answer matrix

        for k in range(0,r1): #* Iterate through 1st matrix rows
            for j in range(0,c2): #* Iterate through 2nd matrix column
                current_sum = 0
                for i in range(0,c1): #* Iterate through first matrix columns
                    
                    current_sum += a.get(k,i)*b.get(i,j)
                
                ans.set(current_sum,k,j);

        return ans
    
    def transpose(self, matrix: Matrix) -> Matrix:

        #* Returns new matrix and updates nothing

        r,c = matrix.size()

        ans = zeros(c,r) #* Create a new matrix with tranpose size
        for i in range(0,r):
            for j in range(0,c):
                ans.set( matrix.get(i,j), j,i )
        
        return ans
    
    def augment_matrices(self, A: Matrix,B: Matrix) -> Matrix:
        
        #* Returns the augmented matrix of A and B
        
        r1,c1 = A.size() #* r1 and r2 should and will be same.
        r2,c2 = B.size()

        augmented_matrix = zeros(r1 , c1+c2) #* Generate enough space for augmented matrix.

        for i in range(0,c1): #* Copy matrix 1.
            augmented_matrix.set_column(i, A.get_column(i))#* Copy matrix 1.

        for i in range(0,c2):
            augmented_matrix.set_column(c1+i, B.get_column(i))#* Copy matrix 2.

        return augmented_matrix

    def solver(self, A: Matrix, B: Matrix) -> Matrix:

        #* Accepts A and B and forms A|B and outputs solution matrix X

        r1,c1 = A.size() #* r1 and r2 should and will be same.
        r2,c2 = B.size()

        augmented_matrix = self.augment_matrices(A,B) #* Generate augmented matrix

        rref_aug = self.rref(augmented_matrix) #* Obtain rref.
        
        c = c1+c2
        r = r1

        #* Back substitution starts here.

        solution_space = zeros(c-1,1) #* Solution

        '''
        Back substitution:

        x(i) = ( t(i) - summation (from j = i+1 to n) { A(i,j)*x(j) } ) / A(i,i)

        Where, x(i) is the final solution value of current pivotal point,

        t(i) is the last element of current row

        A(i,j) is the element of matrix A at i(th) row and j(th) column.

        x(j) is the final solution value of the pivotal point at row j.

        A(i,i) is the element of matrix A at i(th) row and i(th) column.
        
        '''
        current_max_column = 0

        for i in reversed(range(0,r)):
            sum = rref_aug.get(i,c-1) #* Initate sum
            for j in reversed(range(i+1,c)):
                
                if i == j:
                    break
                current_max_column = j
                sum -= solution_space.get(c-j-1,0)*rref_aug.get(i,j)
            sum /= rref_aug.get(i,i) #* sum has solution for current pivotal point

            solution_space.set(sum, c-current_max_column-1, 0)

        return solution_space

class Matrix:

    def __init__(self, values: list[list]) -> None:
        
        if not self._validate_matrixLike(values):
            raise ValueError("Matrix supplied is invalid!");

        self.rows = len(values);
        self.columns = len(values[0]);
        self.matrix = [];

        for i in range(0,self.rows):
            self.matrix.append([]); #* Append rows.

        self.set_matrix(values);

    def __str__(self):
        # Create a string that represents the matrix in the desired format
        result = ""
        for i in range(self.rows):
            for j in range(self.columns):
                result += f"{self.matrix[i][j]}\t"
            result += "\n"
        return result
    
    def size(self) -> tuple[int, int]:

        return (self.rows,self.columns);

    def _validate_matrixLike(self, matrixLike: list[list]) -> bool:
        # Empty input
        if not matrixLike:
            return False
        elif not matrixLike[0]:
            return False;

        no_columns = len(matrixLike[0]);

        for i in range(0, len(matrixLike)): #* Iterate through rows.

            if len(matrixLike[i]) != no_columns: #* Columns aren't uniform
                return False
        
        return True #* Proper input received.

    def set_matrix(self, values: list[list]) -> None:

        #* Expects a 2D list.

        if not self._validate_matrixLike(values):
            raise ValueError("The given input is not a matrix!");

        if len(values) != self.rows:
            raise ValueError("The list passed must satisfy number of rows in matrix.");
        elif len(values[0]) != self.columns:
            raise ValueError("The number of columns must match with given matrix.");

        self.matrix = values; #* Directly force update the values.

    def clear_matrix(self) -> None:
        # Reset matrix
        self.matrix = [];
        
        for i in range(0,self.rows):
            self.matrix.append([]); #* Append rows.
    
    def get(self,r,c):

        if r > self.rows or r < 0:
            raise ValueError("Row out of bound");
        elif c > self.columns or c < 0:
            raise ValueError("Columns out of bound")

        return self.matrix[r][c];

    def _isnumeric(self, value) -> bool:

        if not isinstance(value, (int, float, list)):
            return False
        elif isinstance(value, list):
            if not value:
                return False
            return all(isinstance(x, (int, float)) for x in value)
        return True

    def set(self, value, r,c) -> None:

        if r > self.rows or r < 0:
            raise ValueError("Row out of bound");
        elif c > self.columns or c < 0:
            raise ValueError("Columns out of bound")
        elif not self._isnumeric(value):
            raise ValueError("Invalid value: NaN");
    
        self.matrix[r][c] = value;
    
    def get_row(self, row) -> list:

        if row > self.rows or row < 0:
            raise ValueError("Row out of bound")
        
        return self.matrix[row] #* return row.
    
    def get_column(self, column) -> list:

        if column > self.columns or column < 0:
            raise ValueError("Column out of bound")
        
        ans_column = []
        
        for i in range(0,self.rows):
            ans_column.append(self.matrix[i][column])

        return ans_column #* Return the column
    
    def set_row(self, row: int, value: list[int]) -> None:
        
        if row > self.rows or row < 0:
            raise ValueError("Row out of bound")
        
        if not value or (not isinstance(value, list)):
            raise ValueError("Value is not a valid matrix like")
        
        if len(value) != self.columns:
            raise ValueError("Number of elements in given row is less than other rows.")

        self.matrix[row] = value

    def set_column(self, column: int, value: list[int]) -> None:
        
        if column > self.columns or column < 0:
            raise ValueError("Row out of bound")
        
        if not value or (not isinstance(value, list)):
            raise ValueError("Value is not a valid matrix like")
        
        if len(value) != self.rows:
            raise ValueError("Number of elements in given column is less than other columns.")

        for i in range(0,self.rows):
            self.matrix[i][column] = value[i]

    def det(self) -> int:

        return self._det(self.matrix, 0,0);
    
    def _det(self, matrix, r, c) -> int:
        if len(matrix) == 2 and len(matrix[0]) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        
        ans = 0
        for j in range(len(matrix[0])):
            submatrix = [row[:j] + row[j+1:] for row in matrix[1:]]
            ans += ((-1) ** j) * matrix[0][j] * self._det(submatrix, 0, j)
        
        return ans
    
    #def inv(self) -> Matrix:
        # Returns the inverse (or) adjoint transpose of a matrix.
        
linalg = LinearAlgebra() #* use for quick imports

if __name__ == "__main__":
    print("Error: Cannot run this script!")
    exit(1)