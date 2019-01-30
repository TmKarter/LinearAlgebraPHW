import math

import numpy as np
from fractions import Fraction


def latexCdot(file):
    file.write(" \\cdot ")


def latexMatrix(file, A):
    file.write("\\begin{pmatrix}")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if "<class 'sympy.core.numbers.Rational'>" == str(type(A[i, j])) or "<class 'sympy.core.numbers.Half'>" == str(type(A[i, j])):
                file.write(latexFrac(Fraction(str(A[i, j]))))
            elif not isinstance(A[i, j], str):
                file.write(str(int(A[i, j])))
            else:
                file.write(A[i, j])
            if j != A.shape[1] - 1:
                file.write(" & ")
        if i != A.shape[0] - 1:
            file.write("\\\\")
        file.write("\n")
    file.write("\\end{pmatrix}")


def latexMatrixSquared(file, A):
    latexMatrix(file, A)
    file.write("^2")


def latexMatrixProduct(file, M):
    for i in range(len(M)):
        latexMatrix(file, M[i])
        if i != len(M) - 1:
            latexCdot(file)


def latexMatrixSingleTranspose(file, A):
    latexMatrix(file, A)
    file.write("^T")


def latexMatrixMultipleTranspose(file, M):
    file.write("\\Biggl[")
    latexMatrixProduct(file, M)
    file.write("\\Biggl]^{T}")


def latexDfrac(ab):
    numerator = ab.numerator
    denominator = ab.denominator
    gcd = math.gcd(ab.numerator, ab.denominator)
    numerator = numerator // gcd
    denominator = denominator // gcd
    if denominator == 1:
        return str(numerator)
    return "\\dfrac{" + str(numerator) + "}{" + str(denominator) + "}"


def latexFrac(ab):
    numerator = ab.numerator
    denominator = ab.denominator
    gcd = math.gcd(ab.numerator, ab.denominator)
    numerator = numerator // gcd
    denominator = denominator // gcd
    if denominator == 1:
        return str(numerator)
    elif denominator == -1:
        return str(-numerator)
    sign = ""
    if numerator * denominator < 0:
        sign = "-"
        numerator = abs(numerator)
        denominator = abs(denominator)
    return sign + "\\nicefrac{" + str(numerator) + "}{" + str(denominator) + "}"


def latexRowVector(file, vector, name="", needEqualSign=True):
    file.write(name + needEqualSign * "=" + "\\left(")
    delim = ""
    for elem in np.array(vector):
        if "<class 'numpy.ndarray'>" == str(type(elem)):
            elem = elem[0]
        file.write(delim + str(elem))
        delim = ", "
    file.write(" \\right)")

def latexPolyVector(file, poly):
    result = ""
    first = True
    for power, coefficient in enumerate(poly):
        if power == 0:
            if coefficient != 0:
                result += str(coefficient)
                first = False
        elif power == 1:
            if coefficient == 0:
                continue
            first = False
            if coefficient < 0:
                if coefficient == -1:
                    coefficient = "-"
                result += str(coefficient)
            else:
                if not first:
                    result += "+"
                if coefficient == 1:
                    coefficient = ""
                result += str(coefficient)
            result += "x"
        else:
            if coefficient == 0:
                continue
            if coefficient < 0:
                if coefficient == -1:
                    coefficient = "-"
                result += str(coefficient)
            else:
                if not first:
                    result += "+"
                if coefficient == 1:
                    coefficient = ""
                result += str(coefficient)
            result += "x^2"
    file.write(result)

def latexWriteSOLE(equations, b, file, sign):
    file.write("$\\begin{cases}")
    for row in range(equations.shape[0]):
        elements = 0
        for column in range(equations.shape[1]):
            if equations[row, column] == 0:
                continue
            if column != 0:
                file.write("+" if (equations[row, column] > 0 and elements != 0) else "")
            if equations[row, column] != 1 and equations[row, column] != -1:
                file.write(str(int(equations[row, column])))
            elif int(equations[row, column]) == -1:
                file.write("-")
            file.write("x_{" + str(column + 1) + "}")
            elements += 1
        file.write(" &= " + str(int(b[row])))
        if row < len(equations) - 1:
            file.write(", \\\\")
        else:
            file.write(sign)
    file.write("\\end{cases}$ \n")
