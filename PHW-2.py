import sys

import numpy as np
import sympy as sp
from numpy import linalg as la

import headers
import latex

sys.setrecursionlimit(15000)

randomSeed = 1488


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


def generateFirstTask():
    global randomSeed
    low = -10
    high = 21
    size = (5, 1)

    firstColumn = randomColumn(low, high, size, False, 5)
    secondColumn = randomColumn(low, high, size, True, 4)
    while la.matrix_rank(np.concatenate((firstColumn, secondColumn), 1)) == 1:
        secondColumn = randomColumn(low, high, size, True, 4)
    thirdColumn = randomColumn(low, high, size, True, 4)
    while la.matrix_rank(np.concatenate((firstColumn, secondColumn, thirdColumn), 1)) == 2:
        thirdColumn = randomColumn(low, high, size, True, 4)

    low = -3
    high = 7
    size = 3

    alpha = randomColumn(low, high, size, False, 2)
    beta = randomColumn(low, high, size, False, 2)

    fourthColumn = firstColumn * alpha[0] + secondColumn * alpha[1] + thirdColumn * alpha[2]
    fifthColumn = firstColumn * beta[0] + secondColumn * beta[1] + thirdColumn * beta[2]

    matrix = np.matrix(np.concatenate((firstColumn, secondColumn, thirdColumn, fourthColumn, fifthColumn), axis=1))

    RREF = np.matrix(sp.Matrix(matrix).rref()[0])
    for row in RREF:
        for element in np.array(row)[0]:
            if "<class 'sympy.core.numbers.Rational'>" == str(type(element)) or "<class 'sympy.core.numbers.Half'>" == str(type(element)):
                return generateFirstTask()

    first = RREF[0, :].reshape((5, 1))
    second = RREF[1, :].reshape((5, 1))
    third = RREF[2, :].reshape((5, 1))

    alpha = randomColumn(low, high, size, hasZeroes=False, uniqueElements=3)
    matrix = np.matrix(alpha[0] * first + alpha[1] * second + alpha[2] * third).reshape((5, 1))
    for column in range(1, 5):
        alpha = randomColumn(low, high, size, hasZeroes=False, uniqueElements=3)
        matrix = np.c_[matrix, alpha[0] * first + alpha[1] * second + alpha[2] * third]

    for row in matrix:
        for element in np.array(row)[0]:
            if abs(element) > 25:
                return generateFirstTask()

    np.random.seed(randomSeed)
    randomSeed += 1
    np.random.shuffle(matrix)

    RREF = np.matrix(sp.Matrix(matrix).rref()[0])
    for row in RREF:
        for element in np.array(row)[0]:
            if "<class 'sympy.core.numbers.Rational'>" == str(type(element)) or "<class 'sympy.core.numbers.Half'>" == str(type(element)):
                return generateFirstTask()

    return matrix


def writeFirstTask(tasksFile, answersFile):
    matrix = generateFirstTask()

    tasksFile.write("{\\noindent \\bf 1.} "
                    "Представьте матрицу\n"
                    "\\[ A = ")
    latex.latexMatrix(tasksFile, matrix)
    tasksFile.write("\\]\n"
                    "в виде суммы $r$ матриц ранга 1, где $r = \\operatorname{rk} A$.")

    RREF = np.matrix(sp.Matrix(matrix).rref()[0])

    answersFile.write("{\\noindent \\bf 1.} "
                      "Приведём матрицу к УСВ: \n"
                      "\\[ A' = ")
    latex.latexMatrix(answersFile, RREF)
    answersFile.write(".\\]")


def generateSecondTask():
    global randomSeed
    low = -4
    high = 5
    size = (4, 1)

    u1 = randomColumn(low, high, size, False, 4)
    u2 = randomColumn(low, high, size, False, 4)
    while la.matrix_rank(np.concatenate((u1, u2), 1)) == 1:
        u2 = randomColumn(low, high, size, False, 4)

    alpha, beta = randomNumbers(-3, 4, 2)
    while abs(alpha * beta) < 2:
        alpha, beta = randomNumbers(-3, 4, 2)

    u3 = alpha * u1 + beta * u2

    originalU1, originalU2 = u1, u2

    alpha, beta, gamma, delta = randomNumbers(-7, 7, 4)

    testForSimpleSolution = False
    for k1 in [-1, 0, 1]:
        for k2 in [-1, 0, 1]:
            if (k1 * (alpha * u1 + beta * u2) + k2 * (gamma * u1 + delta * u2) == u3).all():
                testForSimpleSolution = True

    while (abs(alpha * beta) < 2 and abs(gamma * delta) < 2 or
           la.matrix_rank(np.c_[alpha * u1 + beta * u2, gamma * u1 + delta * u2]) != 2 or
           testForSimpleSolution):
        alpha, beta, gamma, delta = randomNumbers(-7, 7, 4)
        testForSimpleSolution = False
        for k1 in [-1, 0, 1]:
            for k2 in [-1, 0, 1]:
                if (k1 * (alpha * u1 + beta * u2) + k2 * (gamma * u1 + delta * u2) == u3).all():
                    testForSimpleSolution = True
    u1, u2 = alpha * u1 + beta * u2, gamma * u1 + delta * u2

    low = -11
    high = 12
    size = (4, 1)

    u4 = randomColumn(low, high, size, False, 4)
    while la.matrix_rank(np.c_[u1, u2, u3, u4]) == 2:
        u4 = randomColumn(low, high, size, False, 4)

    if (len(np.unique([elem for elem in u1])) < 3 and
            len(np.unique([elem for elem in u2])) < 3 or
            len(np.unique([elem for elem in u3])) < 3 and
            len(np.unique([elem for elem in u4])) < 3 or
            len({tuple(abs(elem) for elem in np.array(row)[0]) for row in [u1, u2, u3, u4]}) < 4):
        return generateSecondTask()

    alpha, beta = randomNumbers(-6, 8, 2)
    while (abs(alpha * beta) < 2 or max([abs(elem) for elem in alpha * originalU1 + beta * originalU2]) > 25 or
           la.matrix_rank(np.c_[alpha * originalU1 + beta * originalU2, u1]) == 1 or
           la.matrix_rank(np.c_[alpha * originalU1 + beta * originalU2, u2]) == 1 or
           la.matrix_rank(np.c_[alpha * originalU1 + beta * originalU2, u3]) == 1):
        alpha, beta = randomNumbers(-6, 8, 2)

    v1 = alpha * originalU1 + beta * originalU2

    v2 = randomColumn(low, high, size, False, 4)
    while la.matrix_rank(np.c_[u1, u2, u3, u4, v2]) != 4:
        v2 = randomColumn(low, high, size, False, 4)

    shuffle = False
    if randomNumbers(0, 2, 1) == 0:
        v1, v2 = v2, v1
        shuffle = True

    countZeros = 0
    countDoubles = 0
    for vector in [u1, u2, u3, u4, v1, v2]:
        for elem in np.array(vector):
            if elem == 0:
                countZeros += 1
            if abs(elem) >= 10:
                countDoubles += 1

    if countZeros >= 2 or countDoubles >= 5:
        return generateSecondTask()

    matrix = np.matrix(np.column_stack((u1, u2, u3, u4)))
    RREF = np.matrix(sp.Matrix(matrix).rref()[0])
    for row in RREF:
        for element in np.array(row)[0]:
            if "<class 'sympy.core.numbers.Rational'>" == str(type(element)) or "<class 'sympy.core.numbers.Half'>" == str(type(element)):
                return generateSecondTask()

    return u1, u2, u3, u4, v1, v2, shuffle


def writeSecondTask(tasksFile, answersFile):
    u1, u2, u3, u4, v1, v2, shuffle = generateSecondTask()

    tasksFile.write("{\\noindent \\bf 2.} "
                    "Пусть $U$ --- подпространство в $\R^4$, порождённое векторами \n"
                    "\\["
                    "u_1 = ")
    latex.latexMatrix(tasksFile, u1)
    tasksFile.write(", \n u_2 = ")
    latex.latexMatrix(tasksFile, u2)
    tasksFile.write(", \n u_3 = ")
    latex.latexMatrix(tasksFile, u3)
    tasksFile.write(", \n u_4 = ")
    latex.latexMatrix(tasksFile, u4)
    tasksFile.write(".\\] \n"
                    "Докажите, что из двух векторов \n"
                    "\\["
                    "v_1 = ")
    latex.latexMatrix(tasksFile, v1)
    tasksFile.write(", \n v_2 = ")
    latex.latexMatrix(tasksFile, v2)
    tasksFile.write("\\] \n ровно один лежит в $U$, и дополните этот вектор до базиса подпространства $U$. \n")


groupsSize = 60
groupsNumber = 9

for index in range(3, groupsNumber + 1):
    # Создание файлов для записи условий и ответов, подстановка номера группы в имена файлов
    tasksFile = open("18" + str(index) + "_tasks.tex", 'w')
    answersFile = open("18" + str(index) + "_answers.tex", 'w')

    headers.latexHeader(tasksFile)
    headers.latexHeader(answersFile)

    for i in range((index - 1) * groupsSize + 1, index * groupsSize + 1):
        group = 180 + int((i - 1) / groupsSize) + 1
        variant = i - groupsSize * int((i - 1) / groupsSize)

        headers.tasksHeader(tasksFile, group, variant)
        headers.answersHeader(answersFile, group, variant)

        # writeFirstTask(tasksFile, answersFile)
        writeSecondTask(tasksFile, answersFile)

        tasksFile.write("\n \\newpage\n")
        answersFile.write("\n \\newpage\n")

    tasksFile.write("\n \\end{document}")
    answersFile.write("\n \\end{document}")
    tasksFile.close()
    answersFile.close()
