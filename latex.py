from fractions import Fraction
import math


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
