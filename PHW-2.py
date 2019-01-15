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


def generateThirdTask():
    global randomSeed
    low = -5
    high = 5
    size = (1, 4)

    a1 = randomNumbers(low, high, size)
    a2 = randomNumbers(low, high, size)
    a1[0, 1] = 0
    a2[0, 2] = 0  # линейная независимость а1 и а2 почтинаверное
    a3 = 3 * a1 - 2 * a2
    a4 = 2 * a1 - 3 * a2
    mat_a = np.concatenate((a1, a2, a3, a4), axis=0)
    # print(mat_a)
    rank_l1 = la.matrix_rank(mat_a)
    mat_b = np.zeros((4, 4))
    mat_c = np.zeros((5, 4))
    rank_l2 = rank_u = rank_w = -1
    # в зависимости от seed генерим 3 случая
    var = randomNumbers(low, high, 1)
    if (var % 3 == 0):
        b1 = randomNumbers(low, high, size)
        b2 = 2 * a1 - a2
        b3 = 2 * b1 - b2
        b4 = b1 - 2 * b2
        mat_b = np.concatenate((b1, b2, b3, b4), axis=0)
        rank_l2 = la.matrix_rank(mat_b)
        mat_c = np.concatenate((a1, a2, b1, b2, b3), axis=0)
        rank_u = la.matrix_rank(mat_c)
        rank_w = 2 + rank_l2 - rank_u
    elif (var % 3 == 1):
        b1 = randomNumbers(low, high, size)
        b2 = randomNumbers(low, high, 1) * a1 + randomNumbers(low, high, 1) * a2
        b3 = a1 + randomNumbers(low, high, 1) * a2
        b4 = b1 - 2 * b2
        mat_b = np.concatenate((b1, b2, b3, b4), axis=0)
        rank_l2 = la.matrix_rank(mat_b)
        mat_c = np.concatenate((a1, a2, b1, b2, b3), axis=0)
        rank_u = la.matrix_rank(mat_c)
        rank_w = 2 + rank_l2 - rank_u
    else:
        b1 = randomNumbers(low, high, 1) * a2 + randomNumbers(low, high, 1) * a1
        b2 = randomNumbers(low, high, 1) * a1 + a2
        b3 = 2 * b1 - b2
        b4 = b1 - 2 * b2
        mat_b = np.concatenate((b1, b2, b3, b4), axis=0)
        rank_l2 = la.matrix_rank(mat_b)
        mat_c = np.concatenate((a1, a2, b1, b2, b3), axis=0)
        rank_u = la.matrix_rank(mat_c)
        rank_w = 2 + rank_l2 - rank_u
    return mat_a, mat_b, rank_l1, rank_l2, rank_u, rank_w


def writeThirdTask(tasksFile, answersFile):
    mat_a, mat_b, rank_l1, rank_l2, rank_u, rank_w = generateThirdTask()

    tasksFile.write("{\\noindent \\bf 3.} "
                    "Найдите базис и размерность каждого из подпространств $L_1,\\ L_2,\\ U = L_1 + L_2,\\ W = L_1 \\cap L_2$ пространства $\\R^4$, если $L_1$~--- линейная оболочка векторов\n")
    tasksFile.write("\\["
                    "a_1 = (" + str(mat_a[0, 0]) + ", " + str(mat_a[0, 1]) + ", " + str(mat_a[0, 2]) + ", " + str(mat_a[0, 3]) + ")")
    tasksFile.write(", \\quad a_2 = (" + str(mat_a[1, 0]) + ", " + str(mat_a[1, 1]) + ", " + str(mat_a[1, 2]) + ", " + str(mat_a[1, 3]) + ")")
    tasksFile.write(", \\quad a_3 = (" + str(mat_a[2, 0]) + ", " + str(mat_a[2, 1]) + ", " + str(mat_a[2, 2]) + ", " + str(mat_a[2, 3]) + ")")
    tasksFile.write(", \\quad a_4 = (" + str(mat_a[3, 0]) + ", " + str(mat_a[3, 1]) + ", " + str(mat_a[3, 2]) + ", " + str(mat_a[3, 3]) + ")")
    tasksFile.write(", \\] \n ")
    tasksFile.write("a $L_2$~--- линейная оболочка векторов\n")
    tasksFile.write("\\["
                    "b_1 = (" + str(mat_b[0, 0]) + ", " + str(mat_b[0, 1]) + ", " + str(mat_b[0, 2]) + ", " + str(mat_b[0, 3]) + ")")
    tasksFile.write(", \\quad b_2 = (" + str(mat_b[1, 0]) + ", " + str(mat_b[1, 1]) + ", " + str(mat_b[1, 2]) + ", " + str(mat_b[1, 3]) + ")")
    tasksFile.write(", \\quad b_3 = (" + str(mat_b[2, 0]) + ", " + str(mat_b[2, 1]) + ", " + str(mat_b[2, 2]) + ", " + str(mat_b[2, 3]) + ")")
    tasksFile.write(", \\quad b_4 = (" + str(mat_b[3, 0]) + ", " + str(mat_b[3, 1]) + ", " + str(mat_b[3, 2]) + ", " + str(mat_b[3, 3]) + ")")
    tasksFile.write(".\\] \n")

    answersFile.write("{\\noindent \\bf 3.} ")
    answersFile.write("dim $L_1$ = " + str(rank_l1) + ", dim $L_2$ = " + str(rank_l2) + ", dim U = " + str(rank_u) + ", dim W = " + str(rank_w) + ".\n")


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
        # writeSecondTask(tasksFile, answersFile)
        writeThirdTask(tasksFile, answersFile)

        tasksFile.write("\n \\newpage\n")
        answersFile.write("\n \\newpage\n")

    tasksFile.write("\n \\end{document}")
    answersFile.write("\n \\end{document}")
    tasksFile.close()
    answersFile.close()
    
def generateFourthTask():
    global randomSeed
    low = -10
    high = 10
    size = 1
    
    p1 = randomNumbers(low, high, 1)
    while (p1 == 0):
        p1 = randomNumbers(low, high, 1)
    p2 = randomNumbers(low, high, 1)
    while (p2 == 0 or p1 == p2):
        p2 = randomNumbers(low, high, 1)
    q1 = randomNumbers(low, high, 1)
    while (q1 == 0 or p1 == q1):
        q1 = randomNumbers(low, high, 1)
    q2 = randomNumbers(low, high, 1)
    while (q2 == 0 or q1 == q2 or q2 == p2):
        q2 = randomNumbers(low, high, 1)
    mat_cur = np.array([[1,0,-p1,-q1],[0,0,0,0],[0,1,-p2,-q2],[0,0,0,0]])
    mat_rand = randomNumbers(-41, 41, (4, 4))
    mat_u = np.dot(mat_rand, mat_cur)
    mat_ans = np.array([[*p1, p2, 1, 0],[q1,q2,0,1]])
    return mat_ans, mat_u

def writeThirdTask(tasksFile, answersFile):
    mat_ans, mat_u = generateFourthTask()
    tasksFile.write("{\\noindent \\bf 4.} "
                    "Пусть $U$~---подпространство в $R^4$, натянутое на векторы\n")
    tasksFile.write("\\["
                    "u_1 = (" + str(mat_u[0, 0]) + ", " + str(mat_u[0, 1]) + ", " + str(mat_u[0, 2]) + ", " + str(mat_u[0, 3]) + ")")
    tasksFile.write(", \\quad u_2 = (" + str(mat_u[1, 0]) + ", " + str(mat_u[1, 1]) + ", " + str(mat_u[1, 2]) + ", " + str(mat_u[1, 3]) + ")")
    tasksFile.write(", \\quad u_3 = (" + str(mat_u[2, 0]) + ", " + str(mat_u[2, 1]) + ", " + str(mat_u[2, 2]) + ", " + str(mat_u[2, 3]) + ")")
    tasksFile.write(", \\quad u_4 = (" + str(mat_u[3, 0]) + ", " + str(mat_u[3, 1]) + ", " + str(mat_u[3, 2]) + ", " + str(mat_u[3, 3]) + ")")
    tasksFile.write(", \\] \n ")
    tasksFile.write("Составьте однородную систему линейных уравнений, у которой множество решений совпадает с $U$")
    answersFile.write("{\\noindent \\bf 4.} ")
    answersFile.write(" Общий метод решения выглядит так: \n"
                      "1. Составляется однородная система, строками матрицы которой являются координаты данных векторов \n"
                      "2. Находится её фундаментальное решение \n"
                      "3. Матрицей искомой однородной системы является матрица, строками которой являются векторы полученного фундаментального решения \n"
                      "Итоговая система: \n"
                      "\\[\\begin{cases}")
    answersFile.write(str(mat_ans[0,0]) + "$x_1$" + "+" + str(mat_ans[0,1]) +  "$x_2$" + "+" + "$x_3$ = 0" + "\\\\")
    answersFile.write(str(mat_ans[1,0]) + "$x_1$" + "+" + str(mat_ans[1,1]) + "$x_2$" + "+" + "$x_4$ = 0")
    answersFile.write("\\end{cases}\\]\n")
    answersFile.write("Конечно, студенты могут предъявить т какую-то другую СЛУ, порожденную данным базисом.\n"
                      "Однако, вот эта СЛУ --- самый очевидный и естественный ответ.\n")
