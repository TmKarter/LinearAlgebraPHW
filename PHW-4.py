import sys

import numpy as np
import sympy as sp
from fractions import Fraction
from numpy import linalg as la

import headers
import latex

sys.setrecursionlimit(15000)

randomSeed = 186


def randomNumbers(low, high, size):
    global randomSeed
    np.random.seed(randomSeed)
    randomSeed += 1
    return np.random.randint(low, high, size)


def randomColumn(low, high, size, hasZeroes=True, uniqueElements=1):
    column = randomNumbers(low, high, size)
    while (not hasZeroes and 0 in column) or (len(np.unique(column)) < uniqueElements):
        column = randomNumbers(low, high, size)
    return column


def generateNewBasis(basis, coefficients):
    new = True
    newBasis = []
    for column in range(coefficients.shape[1]):
        vector = []
        newVector = True
        for row in range(coefficients.shape[0]):
            if newVector:
                vector = coefficients[row, column] * basis[row, :]
                newVector = False
            else:
                vector += coefficients[row, column] * basis[row, :]
        if new:
            newBasis = np.matrix(vector).T
            new = False
        else:
            newBasis = np.c_[newBasis, vector]
    return newBasis


def generateTask3():
    low = -3
    high = 4
    size = (3, 3)
    basis = np.zeros(size)
    while (la.matrix_rank(basis) != 3 or
           len({abs(elem) for elem in np.unique(basis)}) < 3 or
           0 in np.unique(basis)):
        basis = randomNumbers(low, high, size)

        GCDs = []
        for i in range(3):
            GCDs.append(np.gcd.reduce(basis[i, :]))
        if 0 not in GCDs:
            basis = (basis.T / GCDs).T.astype(int)

        GCDs = []
        for i in range(3):
            GCDs.append(np.gcd.reduce(basis.T[i, :]))
        if 0 not in GCDs:
            basis = (basis / GCDs).astype(int)

    low = -2
    high = 3
    size = (3, 3)
    coefficients = randomNumbers(low, high, size)
    bad = True
    while bad or la.det(coefficients) == 0:
        coefficients = randomNumbers(low, high, size)
        bad = False
        for row in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]:
            if any((row == x).all() for x in coefficients) or any((row == x).all() for x in coefficients.T):
                bad = True
        if la.det(coefficients) != 0:
            inversedCoeffients = np.matrix(sp.Matrix(coefficients).inv())
            for row in inversedCoeffients:
                row = np.array(row)[0]
                for elem in row:
                    # print("\n !!!! \n", elem, "=", type(elem), "\n !!!! \n")
                    if "<class 'sympy.core.numbers.Rational'>" == str(type(elem)) and abs(Fraction(str(elem)).denominator) > 4:
                        bad = True
        else:
            bad = True
    inversedCoeffients = np.matrix(sp.Matrix(coefficients).inv())
    newBasis = generateNewBasis(basis, coefficients)

    low = -5
    high = 6
    size = (1, 3)
    vector = randomNumbers(low, high, size)
    while len({abs(elem) for elem in np.unique(vector)}) < 3 or 0 in vector:
        vector = randomNumbers(low, high, size)

    vectorInNewBasis = inversedCoeffients.dot(vector.T).T
    return basis, newBasis, coefficients, inversedCoeffients, vector, vectorInNewBasis


def writeTask3(tasksFile, answersFile):
    basis, newBasis, coefficients, inversedCoeffients, vector, vectorInNewBasis = generateTask3()

    tasksFile.write("{\\noindent \\bf 3.} "
                    "В пространстве $\\R^3$ заданы два базиса "
                    "$\\bbm{e}=(e_1,e_2,e_3)$ и $\\bbm{e}' = (e'_1,e'_2,e'_3)$, где "
                    "\\[")
    latex.latexRowVector(tasksFile, basis[:, 0], "e_1")
    tasksFile.write(", \\quad ")
    latex.latexRowVector(tasksFile, basis[:, 1], "e_2")
    tasksFile.write(", \\quad ")
    latex.latexRowVector(tasksFile, basis[:, 2], "e_3")
    tasksFile.write(", \\qquad ")
    latex.latexRowVector(tasksFile, newBasis[:, 0], "e'_1")
    tasksFile.write(", \\quad ")
    latex.latexRowVector(tasksFile, newBasis[:, 1], "e'_2")
    tasksFile.write(", \\quad ")
    latex.latexRowVector(tasksFile, newBasis[:, 2], "e'_3")
    tasksFile.write(",\\]"
                    "и вектор $v$, имеющий в базисе $\\bbm{e}$ координаты $")
    latex.latexRowVector(tasksFile, vector.T, "", needEqualSign=False)
    tasksFile.write("$. Найдите:"
                    "\\begin{itemize}\n"
                    "\\item[(a)] матрицу перехода от базиса $\\bbm{e}$ к базису $\\bbm{e}'$; "
                    "\\item[(б)] координаты вектора $v$ в базисе $\\bbm{e}'$.\\end{itemize}\n")
    tasksFile.write("\n \\medskip \n")

    answersFile.write("{\\noindent \\bf 3.} "
                      "Матрицы смены базиса: \\[ T_{\\bbm{e} \\to \\bbm{e'}} = ")
    latex.latexMatrix(answersFile, coefficients)
    answersFile.write(", \\qquad T_{\\bbm{e'} \\to \\bbm{e}} = ")
    latex.latexMatrix(answersFile, inversedCoeffients)
    answersFile.write(".\\]\n")
    answersFile.write("Координаты вектора $v$ в новом базисе: $")
    latex.latexRowVector(answersFile, vectorInNewBasis.T, name="", needEqualSign=False)
    answersFile.write("$. \\\\ \\\\")
    tasksFile.write("\n\\medskip\n")



def generateTask4():
    low = -5
    high = 6
    size = (3, 3)

    basisPoly = np.zeros(size)
    while (abs(la.det(basisPoly)) != 2 or
           len({abs(elem) for elem in np.unique(basisPoly)}) < 3 or
           0 in np.unique(basisPoly)):
        basisPoly = randomNumbers(low, high, size)

        GCDs = []
        for i in range(3):
            GCDs.append(np.gcd.reduce(basisPoly[i, :]))
        if 0 not in GCDs:
            basisPoly = (basisPoly.T / GCDs).T.astype(int)

        GCDs = []
        for i in range(3):
            GCDs.append(np.gcd.reduce(basisPoly.T[i, :]))
        if 0 not in GCDs:
            basisPoly = (basisPoly / GCDs).astype(int)

    low = -4
    high = 5
    size = (2, 2)
    basisR2 = np.zeros(size)
    while (la.matrix_rank(basisR2) != 2 or
           len({abs(elem) for elem in np.unique(basisR2)}) < 2 or
           0 in np.unique(basisR2)):
        basisR2 = randomNumbers(low, high, size)

        GCDs = []
        for i in range(2):
            GCDs.append(np.gcd.reduce(basisR2[i, :]))
        if 0 not in GCDs:
            basisR2 = (basisR2.T / GCDs).T.astype(int)

        GCDs = []
        for i in range(2):
            GCDs.append(np.gcd.reduce(basisR2.T[i, :]))
        if 0 not in GCDs:
            basisR2 = (basisR2 / GCDs).astype(int)

    low = -7
    high = 8
    size = (2, 3)
    mapMatrix = np.zeros(size)
    while (len({abs(elem) for elem in np.unique(mapMatrix)}) < 5 or
           0 in np.unique(mapMatrix)):
        mapMatrix = randomNumbers(low, high, size)

        GCDs = []
        for i in range(2):
            GCDs.append(np.gcd.reduce(mapMatrix[i, :]))
        if 0 not in GCDs:
            mapMatrix = (mapMatrix.T / GCDs).T.astype(int)

        GCDs = []
        for i in range(3):
            GCDs.append(np.gcd.reduce(mapMatrix.T[i, :]))
        if 0 not in GCDs:
            mapMatrix = (mapMatrix / GCDs).astype(int)

    low = -2
    high = 3
    size = (3, 1)
    coefficientsPoly = randomNumbers(low, high, size)
    while len({abs(elem) for elem in np.unique(coefficientsPoly)}) < 2 or 0 in coefficientsPoly:
        coefficientsPoly = randomNumbers(low, high, size)

    poly = coefficientsPoly[0][0] * basisPoly[0, :] + coefficientsPoly[1][0] * basisPoly[1, :] + coefficientsPoly[2][0] * basisPoly[2, :]
    coefficientsR2 = mapMatrix.dot(coefficientsPoly)
    mappedPoly = basisR2[0, :] * coefficientsR2[0][0] + basisR2[1, :] * coefficientsR2[1][0]
    return basisPoly, basisR2, mapMatrix, poly, mappedPoly, coefficientsPoly, coefficientsR2


def writeTask4(tasksFile, answersFile):
    basisPoly, basisR2, mapMatrix, poly, mappedPoly, coefficientsPoly, coefficientsR2 = generateTask4()
    tasksFile.write("{\\noindent \\bf 4.} "
                    "Пусть $V = \\R[x]_{\\leqslant 2}$ --- пространство многочленов с "
                    "действительными коэффициентами от переменной $x$ степени не выше 2. "
                    "Линейное отображение $\\varphi \\colon V \\to \\R^2$ в базисе $(")
    latex.latexPolyVector(tasksFile, basisPoly[0, :])
    tasksFile.write(",")
    latex.latexPolyVector(tasksFile, basisPoly[1, :])
    tasksFile.write(",")
    latex.latexPolyVector(tasksFile, basisPoly[2, :])
    tasksFile.write(")$ пространства $V$ и базисе $(")
    latex.latexRowVector(tasksFile, basisR2[0, :], "", False)
    tasksFile.write(", ")
    latex.latexRowVector(tasksFile, basisR2[1, :], "", False)
    tasksFile.write("$) пространства $\\R^2$ имеет матрицу \\[")
    latex.latexMatrix(tasksFile, mapMatrix)
    tasksFile.write(".\\] Найдите $\\varphi(")
    latex.latexPolyVector(tasksFile, poly)
    tasksFile.write(")$.")

    answersFile.write("{\\noindent \\bf 4.} "
                      "Коэффициенты многочлена в $V$: $")
    latex.latexRowVector(answersFile, coefficientsPoly, "", False)
    answersFile.write("$. \\\\ \n"
                      "Коэффициенты многочлена при отображении: $")
    latex.latexRowVector(answersFile, coefficientsR2, "", False)
    answersFile.write("$. \\\\ \n"
                      "Сам вектор в $\\R^2$: $")
    latex.latexRowVector(answersFile, mappedPoly, "", False)
    answersFile.write("$. \\\\ \\\\")
    tasksFile.write("\n\\medskip\n")


groupsSize = 45
groupsNumber = 9

for index in range(3, groupsNumber + 1):
    if index == 4:
        continue

    tasksFile = open("18" + str(index) + "_tasks4.tex", 'w')
    answersFile = open("18" + str(index) + "_answers4.tex", 'w')

    headers.latexHeader(tasksFile)
    headers.latexHeader(answersFile)

    for i in range((index - 1) * groupsSize + 1, index * groupsSize + 1):
        group = 180 + int((i - 1) / groupsSize) + 1
        variant = i - groupsSize * int((i - 1) / groupsSize)

        headers.tasksHeader(tasksFile, group, variant)
        headers.answersHeader(answersFile, group, variant)

        answersFile.write("Во всех задачах намеренно не приводится улучшенный ступенчатый вид --- "
                          "те матрицы, которые дают библиотеки, и те матрицы, которые найдут студенты, сильно разнятся, "
                          "ввиду чего смысл этого действа пропадает. \\\\ \\\\ \n")

        print("Group №" + str(group) + ".", end=' ')
        print("Variant №" + str(variant) + ":")
        # writeFirstTask(tasksFile, answersFile)
        # print("First task was generated.")
        # writeSecondTask(tasksFile, answersFile)
        # print("Second task was generated.")
        writeTask3(tasksFile, answersFile)
        print("Third task was generated.")
        writeTask4(tasksFile, answersFile)
        print("Fourth task was generated.")
        print()

        tasksFile.write("\n \\newpage\n")
        answersFile.write("\n \\newpage\n")

    tasksFile.write("\n \\end{document}")
    answersFile.write("\n \\end{document}")
    tasksFile.close()
    answersFile.close()
