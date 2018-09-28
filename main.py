import math

import numpy as np
import sympy as sp
from fractions import Fraction


def latexHeader(file):
    file.write("\\documentclass{article}\n"
               "\\usepackage{latexsym,amsxtra,amscd,ifthen}\n"
               "\\usepackage{amsfonts}\n"
               "\\usepackage{verbatim}\n"
               "\\usepackage{amsmath}\n"
               "\\usepackage{amsthm}\n"
               "\\usepackage{amssymb}\n"
               "\\usepackage[inline, shortlabels]{enumitem}\n"
               "\\usepackage[russian]{babel}\n"
               "\\usepackage[utf8]{inputenc}\n"
               "\\usepackage[T2A]{fontenc}\n"
               "\\usepackage{nicefrac}\n"
               "\\numberwithin{equation}{section}\n"
               "\\pagestyle{plain}\n"
               "\\textwidth=19.0cm\n"
               "\\oddsidemargin=-1.3cm\n"
               "\\textheight=26cm\n"
               "\\topmargin=-3.0cm\n"
               "\\tolerance=500\n"
               "\\unitlength=1mm\n"
               "\\def\\R{{\\mathbb{R}}}\n"
               "\\begin{document}\n")


def tasksHeader(file, group, variant):
    group = str(group)
    variant = str(variant)
    file.write("\\begin{center}\n"
               "\\footnotesize\n"
               "\\noindent\\makebox[\\textwidth]{Линейная алгебра и геометрия \\hfill ФКН НИУ ВШЭ, 2018/2019 учебный год, 1-й курс ОП ПМИ, основной поток}\n"
               "\\end{center}\n"
               "\\begin{center}\n"
               "\\textbf{Индивидуальное домашнее задание 1}\n"
               "\\end{center}\n"
               "\\begin{center}\n"
               "{Группа БПМИ" + group + ". Вариант " + variant + "}\n\\end{center}\n")


def answersHeader(file, group, variant):
    group = str(group)
    variant = str(variant)
    file.write("\\begin{center}"
               "\\bf Ответы к индивидуальному домашнему заданию 1"
               "\\end{center}"
               "\\begin{center}"
               "{Группа БПМИ" + group + ". Вариант " + variant + "}\n\\end{center}\n")


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
    tasksFile.write("\\Biggl]^{T}")


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
    return "\\nicefrac{" + str(numerator) + "}{" + str(denominator) + "}"


def generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, rows, columns, variant):
    count = 1
    np.random.seed(variant + count)
    count += 1
    matrix = np.random.randint(lowest_bound, highest_bound, (rows, columns))
    minuses = 0
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            while matrix[row][column] == 0 or minuses > (rows * columns) // 3 and matrix[row][column] < 0:
                np.random.seed(variant + count)
                count += 1
                matrix[row][column] = np.random.randint(lowest_bound, highest_bound)
            if matrix[row][column] < 0:
                minuses += 1
            np.random.seed(variant + count)
            count += 1
    numbers = set()
    for row in range(rows):
        for column in range(columns):
            numbers.add(abs(matrix[row][column]))
    if len(numbers) <= 2 * rows * columns // 3:
        return generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, rows, columns, variant + 13)
    return np.matrix(matrix)


def generateFirstTask(variant):
    lowest_bound = -6
    highest_bound = 7

    u_lowest_bound = 1
    u_highest_bound = 6

    v_lowest_bound = 1
    v_highest_bound = 6

    count = 2
    A = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 3, variant * count)
    count += 1
    B = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 3, variant * count)
    count += 1
    C = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 2, variant * count)
    count += 1
    D = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 2, variant * count)
    count += 1

    while np.array_equal(C * D, D * C):
        C = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 2, variant * count)
        count += 1

    u = 0
    while u == -1 or u == 0 or u == 1:
        np.random.seed(variant * count)
        u = np.random.randint(u_lowest_bound, u_highest_bound)
        count += 1
        np.random.seed(variant * count)
        if np.random.randint(0, 3) == 0:
            u *= -1
        count += 1

    v = 0
    while v == -1 or v == 0 or v == 1:
        np.random.seed(variant * count)
        v = np.random.randint(v_lowest_bound, v_highest_bound)
        count += 1
        np.random.seed(variant * count)
        if np.random.randint(0, 3) == 0:
            v *= -1
        count += 1
    return A, B, C, D, u, v


def haveTwoCommonElements(first, second):
    count = 0
    for i in range(len(first)):
        if abs(first[i]) == abs(second[i]):
            count += 1
        if count == 2:
            return True
    return False


def haveCommonSigns(first, second, howMany):
    count = 0
    for i in range(len(first)):
        if first[i] * second[i] > 0:
            count += 1
        if count >= howMany:
            return True
    return False


def haveTwoSimiliarElements(vector):
    for i in range(len(vector)):
        for j in range(i + 1, len(vector)):
            if vector[i] == vector[j]:
                return True
    return False


def hasElementOff(vector, highest):
    for elem in vector:
        if abs(elem) > highest:
            return True
    return False


def writeFirstTask(A, B, C, D, u, v, tasksFile, answers, variant):
    tasksFile.write("{\\noindent \\bf 1.} "
                    "Даны матрицы\n")

    tasksFile.write("\\[\n")
    tasksFile.write("A = ")
    latexMatrix(tasksFile, A)
    tasksFile.write(", B = ")
    latexMatrix(tasksFile, B)
    tasksFile.write(", C = ")
    latexMatrix(tasksFile, C)
    tasksFile.write(", D = ")
    latexMatrix(tasksFile, D)
    tasksFile.write(".\\]\n")
    tasksFile.write("Вычислите матрицу\n")

    tasksFile.write("\\[\n")

    count = 1
    np.random.seed(variant + count)
    swipe_CD1 = np.random.randint(0, 2)
    count += 1

    np.random.seed(variant + count)
    swipe_C1 = np.random.randint(0, 2)
    count += 1

    np.random.seed(variant + count)
    swipe_AB12 = np.random.randint(0, 2)
    count += 1

    tasksFile.write(str(u))
    # Начало первого слагаемого
    first_term = u
    if swipe_C1 and swipe_CD1:
        tasksFile.write("D \\cdot ")
        first_term *= D
    elif swipe_C1:
        tasksFile.write("C \\cdot ")
        first_term *= C

    if swipe_AB12:
        tasksFile.write("A \\cdot A^T")
        first_term *= A * A.transpose()
    else:
        tasksFile.write("B \\cdot B^T")
        first_term *= B * B.transpose()

    if not swipe_C1 and swipe_CD1:
        tasksFile.write(" \\cdot D")
        first_term *= D
    elif not swipe_C1 and not swipe_CD1:
        tasksFile.write(" \\cdot C")
        first_term *= C
    # Конец первого слагаемого

    tasksFile.write("+")
    # tasksFile.write("\\]")
    # tasksFile.write("\\[")
    # tasksFile.write("+")

    # Начало второго слагаемого
    second_term = 1
    tasksFile.write("\\operatorname{tr}\\left(")
    if swipe_AB12:
        tasksFile.write("B^T \\cdot B")
        second_term *= int((B.transpose() * B).trace())
    else:
        tasksFile.write("A^T \\cdot A")
        second_term *= int((A.transpose() * A).trace())
    tasksFile.write("\\right)")
    latexCdot(tasksFile)

    np.random.seed(variant + count)
    swipe_AB2 = np.random.randint(0, 2)
    count += 1

    np.random.seed(variant + count)
    swipe_sign2 = np.random.randint(0, 2)
    count += 1

    first, second = A, B
    sign = "-"
    if swipe_AB2:
        first, second = B, A
    if swipe_sign2:
        sign = "+"
        second_term *= first + second
    else:
        second_term *= first - second

    tasksFile.write("\\left(")
    tasksFile.write("A" if np.array_equal(first, A) else "B")
    tasksFile.write(sign)
    if sign == "+":
        sign = "-"
        second_term = second_term * (first.transpose() - second.transpose())
    else:
        sign = "+"
        second_term = second_term * (first.transpose() + second.transpose())
    tasksFile.write("A" if np.array_equal(second, A) else "B")
    tasksFile.write("\\right)")
    latexCdot(tasksFile)
    tasksFile.write("\\left(")
    tasksFile.write("A^T" if np.array_equal(first, A) else "B^T")
    tasksFile.write(sign)
    tasksFile.write("A^T" if np.array_equal(second, A) else "B^T")
    tasksFile.write("\\right)")
    # Конец второго слагаемого

    # Начало третьего слагаемого
    third_term = v * C * C
    if v > 0:
        tasksFile.write("+ ")
    tasksFile.write(str(v))
    tasksFile.write("C^2")
    # Конец третьего слагаемого

    # Начало четвертого слагаемого
    fourth_term = 1
    np.random.seed(variant + count)
    swipe_sign4 = np.random.randint(0, 2)
    count += 1
    v *= 2
    if swipe_sign4:
        fourth_term *= -v
        if v > 0:
            tasksFile.write("-" + str(v))
        elif v < 0:
            tasksFile.write("+" + str(-1 * v))
    else:
        fourth_term *= v
        if v > 0:
            tasksFile.write("+" + str(v))
        elif v < 0:
            tasksFile.write(str(v))
    tasksFile.write("C \\cdot D")
    fourth_term = C * D * fourth_term
    # Конец четвертого слагаемого

    # Начало пятого слагаемого
    v = int(v // 2)
    fifth_term = v * D * D
    if v > 0:
        tasksFile.write("+")
    tasksFile.write(str(v))
    tasksFile.write("D^2")
    # Конец пятого слагаемого

    tasksFile.write(".\\]\n")
    tasksFile.write("\n \\medskip \n")

    answer = first_term + second_term + third_term + fourth_term + fifth_term
    answers.write("\\[\\] 1. Должна получиться следующая матрица: $$")
    latexMatrix(answers, answer)
    answers.write("$$\n")
    answers.write("\\[\\] Слагаемые слева направо: \\begin{center}\n$$")
    latexMatrix(answers, first_term)
    answers.write(" + ")
    latexMatrix(answers, second_term)
    answers.write(" + ")
    latexMatrix(answers, third_term)
    answers.write(" + ")
    latexMatrix(answers, fourth_term)
    answers.write(" + ")
    latexMatrix(answers, fifth_term)
    answers.write("$$\\end{center}\n")


infinitePrev = []
inconsistentPrev = []
equationsPrev = []


def generateSecondTask(variant):
    numbers_lowest_bound = -9
    numbers_highest_bound = 10

    coefficient_lowest_bound = -5
    coefficient_highest_bound = 6

    count = 9000
    np.random.seed(variant + count)
    first_basis = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))
    count += 1
    while haveTwoSimiliarElements(first_basis):
        np.random.seed(variant + count)
        first_basis = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))
        count += 1

    np.random.seed(variant + count)
    second_basis = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))
    count += 1
    while (np.linalg.matrix_rank(np.matrix(np.column_stack((first_basis, second_basis)))) != 2 or
           haveTwoCommonElements(first_basis, second_basis) or
           haveCommonSigns(first_basis, second_basis, 2) or
           haveTwoSimiliarElements(second_basis)):
        np.random.seed(variant + count)
        second_basis = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))
        count += 1

    third_vector = first_basis
    while (np.array_equal(third_vector, first_basis) or
           np.array_equal(third_vector, second_basis) or
           np.array_equal(third_vector, first_basis + second_basis) or
           np.array_equal(third_vector, first_basis - second_basis) or
           np.array_equal(third_vector, second_basis - first_basis) or
           haveTwoSimiliarElements(third_vector) or
           hasElementOff(third_vector, 27)):
        np.random.seed(variant + count)
        third_vector = np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * first_basis
        count += 1
        np.random.seed(variant + count)
        third_vector += np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * second_basis
        count += 1

    fourth_vector = first_basis
    while (np.array_equal(fourth_vector, first_basis) or
           np.array_equal(fourth_vector, second_basis) or
           np.array_equal(fourth_vector, first_basis + second_basis) or
           np.array_equal(fourth_vector, first_basis - second_basis) or
           np.array_equal(fourth_vector, second_basis - first_basis) or
           np.array_equal(fourth_vector, third_vector) or
           haveTwoCommonElements(third_vector, fourth_vector) or
           haveCommonSigns(third_vector, fourth_vector, 2) or
           haveTwoSimiliarElements(fourth_vector) or
           hasElementOff(fourth_vector, 27)):
        np.random.seed(variant + count)
        fourth_vector = np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * first_basis
        count += 1
        np.random.seed(variant + count)
        fourth_vector += np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * second_basis
        count += 1

    np.random.seed(variant + count)
    column_b_infinite = np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * first_basis
    count += 1
    np.random.seed(variant + count)
    column_b_infinite += np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * second_basis
    count += 1
    found = False
    for vector in infinitePrev:
        if np.array_equal(vector, column_b_infinite):
            found = True
            break
    while found:
        np.random.seed(variant + count)
        column_b_infinite = np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * first_basis
        count += 1
        np.random.seed(variant + count)
        column_b_infinite += np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * second_basis
        count += 1
        found = False
        for vector in infinitePrev:
            if np.array_equal(vector, column_b_infinite):
                found = True
                break
    infinitePrev.append(column_b_infinite)

    np.random.seed(variant + count)
    column_b_inconsistent = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))
    count += 1
    found = False
    for vector in inconsistentPrev:
        if np.array_equal(vector, column_b_inconsistent):
            found = True
            break

    while np.linalg.matrix_rank(np.matrix(np.column_stack((first_basis, second_basis, column_b_inconsistent)))) != 3 or found:
        np.random.seed(variant + count)
        column_b_inconsistent = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))
        count += 1
        found = False
        for vector in inconsistentPrev:
            if np.array_equal(vector, column_b_inconsistent):
                found = True
                break
    inconsistentPrev.append(column_b_inconsistent)
    equations = np.matrix(np.column_stack((first_basis, second_basis, third_vector, fourth_vector)))
    for row in range(len(equations)):
        for column in range(len(equations[row])):
            if equations[row, column] == 0:
                return generateSecondTask(variant + count)
    for matrix in equationsPrev:
        if np.array_equal(equations, matrix):
            return generateSecondTask(variant + count)
    equationsPrev.append(equations)
    return equations, column_b_infinite, column_b_inconsistent


def writeSOLE(equations, b, file, sign):
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
        file.write(" &= " + str(int(b[row][0])))
        if row < len(equations) - 1:
            file.write(", \\\\")
        else:
            file.write(sign)
    file.write("\\end{cases}$ \n \\medskip\n")


def writeSecondTask(equations, infinite, inconsistent, tasksFile, answersFile, variant):
    count = 4000
    np.random.seed(variant + count)
    inconsistentFirst = np.random.randint(0, 2)
    count += 1

    tasksFile.write("{\\noindent \\bf 2.} "
                    "Решите каждую из приведённых ниже систем линейных уравнений методом Гаусса. "
                    "Если система совместна, то выпишите её общее решение и укажите одно частное решение.\\\\")
    equations_infinite = np.c_[equations, infinite]
    equations_inconsistent = np.c_[equations, inconsistent]
    row_reduced_infinite = np.matrix(sp.Matrix(equations_infinite).rref()[0])
    row_reduced_inconsistent = np.matrix(sp.Matrix(equations_inconsistent).rref()[0])
    tasksFile.write("\\begin{center}\n")
    tasksFile.write("(a)\\quad")
    if inconsistentFirst:  # Сначала несовместная система
        writeSOLE(equations, inconsistent, tasksFile, ";")
        answersFile.write("\\[\\] 2. (а) Должна получиться несовместная система со следующей матрицей: $$")
        latexMatrix(answersFile, row_reduced_inconsistent)
        answersFile.write("$$\n")
    else:  # Бесконечное число решений сначале
        writeSOLE(equations, infinite, tasksFile, ";")
        answersFile.write("\\[\\] 2. (а) Должна получиться совместная система с бесконечным количеством решений и следующей матрицей: $$")
        latexMatrix(answersFile, row_reduced_infinite)
        answersFile.write("$$\n")

    tasksFile.write("\\qquad (б)\\quad")

    if inconsistentFirst:  # Бесконечное число решений в конце
        writeSOLE(equations, infinite, tasksFile, ".")
        answersFile.write("\\[\\] (б) Должна получиться совместная система с бесконечным количеством решений и следующей матрицей: $$")
        latexMatrix(answersFile, row_reduced_infinite)
        answersFile.write("$$\n")
    else:  # Несовместная система в конце
        writeSOLE(equations, inconsistent, tasksFile, ".")
        answersFile.write("\\[\\] (б) Должна получиться несовместная система со следующей матрицей: $$")
        latexMatrix(answersFile, row_reduced_inconsistent)
        answersFile.write("$$\n")
    tasksFile.write("\\end{center}\n")
    tasksFile.write("\n \\medskip \n")


def generateThirdTask(variant):
    lowest_bound = -9
    highest_bound = 10
    count = 100
    A = np.zeros((3, 3), int)
    numbers = set()
    for row in range(3):
        for column in range(row, 3):
            while A[row, column] == 0:
                np.random.seed(variant * count + count)
                A[row, column] = np.random.randint(lowest_bound, highest_bound)
                count += 1
            numbers.add(A[row, column])

    minuses = 0
    while len(numbers) <= 5 or minuses > 2 or minuses == 0:
        minuses = 0
        numbers = set()
        for row in range(3):
            for column in range(row, 3):
                np.random.seed(variant * count + count)
                A[row, column] = np.random.randint(lowest_bound, highest_bound)
                count += 1
                while A[row, column] == 0:
                    np.random.seed(variant * count + count)
                    A[row, column] = np.random.randint(lowest_bound, highest_bound)
                    count += 1
                numbers.add(abs(A[row, column]))
                if A[row, column] < 0:
                    minuses += 1

    np.random.seed(variant * count + count)
    pos = np.random.randint(0, 3)
    count += 1
    A[pos, pos] = 0
    possiblePermutations = [np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])]
    np.random.seed(variant * count + count)
    P = possiblePermutations[np.random.randint(0, 4)]
    count += 1
    X = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    return A, P, X


def writeThirdTask(A, P, X, tasksFile, answersFile):
    inverseP = np.transpose(np.matrix(P))
    A = np.matrix(P) * np.matrix(A) * np.matrix(inverseP)
    X = np.matrix(P) * np.matrix(X) * np.matrix(inverseP)
    printableX = np.array([["" for i in range(3)] for j in range(3)])
    for row in range(3):
        for column in range(3):
            if int(X[row, column]) == 1:
                printableX[row, column] = "*"
            else:
                printableX[row, column] = "0"
    tasksFile.write("{\\noindent \\bf 3.} Дана матрица")
    tasksFile.write("\\[\n")
    tasksFile.write("A = ")
    latexMatrix(tasksFile, A)
    tasksFile.write(".\\]\n")
    tasksFile.write("Найдите все матрицы $X$ вида")
    tasksFile.write("\\[\n")
    tasksFile.write("X = ")
    latexMatrix(tasksFile, printableX)
    tasksFile.write(",\\] удовлетворяющие условию $AX = XA.$\n")
    tasksFile.write("\n \\medskip \n")

    answersFile.write("3. Ответ приведен в терминах следующей нумерации переменных матрицы X: $$\n")
    variable = 1
    tempX = [["" for i in range(3)] for j in range(3)]
    for row in range(3):
        for column in range(3):
            if printableX[row, column] == "*":
                tempX[row][column] = "x_" + str(variable)
                variable += 1
            else:
                tempX[row][column] = "0"

    tempX = np.array(tempX)
    latexMatrix(answersFile, tempX)
    answersFile.write("$$\n")

    AX = np.zeros((3, 3, 6), int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if tempX[k, j] == "0":
                    continue
                variable = int(tempX[k, j][2]) - 1
                AX[i, j, variable] = A[i, k]
    XA = np.zeros((3, 3, 6), int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if tempX[i, k] == "0":
                    continue
                variable = int(tempX[i, k][2]) - 1
                XA[i, j, variable] = A[k, j]
    AXminusXA = AX - XA
    equationsThird = []
    for i in range(3):
        for j in range(3):
            if np.array_equal(AXminusXA[i, j], [0, 0, 0, 0, 0, 0]):
                continue
            equationsThird.append(AXminusXA[i, j])
    for i in range(3):
        for j in range(6):
            equationsThird[i][j] = Fraction(equationsThird[i][j], 1)
    equationsThird = np.matrix(equationsThird)
    answersFile.write("После умножения и приравнивания матриц получится следующая однородная система: $$\n")
    latexMatrix(answersFile, equationsThird)
    answersFile.write("$$\n")
    answersFile.write("Таким образом, улучшенный ступенчатый вид этой матрицы выглядит так: $$\n")
    equationsThird = np.matrix(sp.Matrix(equationsThird).rref()[0])
    latexMatrix(answersFile, equationsThird)
    answersFile.write("$$\n")


def generateFourthTask(i):  # Задание выполнено Ильёй Анищенко, большая благодарность ему за помощь
    ran_val_4 = i % 8
    swap_2_3_row = 0
    swap_2_3_col = 0
    ans_ab = Fraction(0, 1)
    ans_a = Fraction(0, 1)
    ans_b = Fraction(0, 1)
    line1 = list()
    line2 = list()
    line3 = list()
    str_1 = ""
    str_2 = ""
    str_3 = ""
    b_part = list()
    ex_cnt = 1
    if (ran_val_4 % 4 == 0):
        swap_2_3_row = 0
        swap_2_3_col = 0
    elif (ran_val_4 % 4 == 1):
        swap_2_3_row = 0
        swap_2_3_col = 1
    elif (ran_val_4 % 4 == 2):
        swap_2_3_row = 1
        swap_2_3_col = 0
    elif (ran_val_4 % 4 == 3):
        swap_2_3_row = 1
        swap_2_3_col = 1

    if (ran_val_4 < 4):  # 1 вариант
        s1 = -((((i ** 4 + 3) % (7 + 11 * i)) % 7) + 1) * (-1) ** ((i + 5) ** 4 % 2)
        s2 = -((((i ** 4 + 5) % (7 + 11 * i)) % 5) + 1) * (-1) ** ((i + 5) ** 2 % 4)
        t1 = 1 * (-1) ** ((i + 5) ** 3 % 3)
        t2 = -((((i ** 3 + 2) % (7 + 12 * i)) % 6) + 1) * (-1) ** ((i + 5) ** 6 % 7)
        t3 = -((((i ** 3 + 1) % (7 + 13 * i)) % 5) + 1) * (-1) ** ((i + 5) ** 5 % 5 + 1)
        u1 = -((((i ** 5 + 2) % (7 + 10 * i)) % 8) + 1) * (-1) ** ((i + 4) ** 3 % 5)
        u2 = -((((i ** 6 + 3) % (11 + 12 * i)) % 9) + 1) * (-1) ** ((i + 7) ** 3 % 3)
        z = -((((i ** 2 + 5) % (13 + 12 * i)) % 7) + 1) * (-1) ** ((i + 4) ** 3 % 6)
        while (s1 * t2 - t1 * s2 == 0):
            t2 = -((((i ** 4 + 3 * ex_cnt) % (17 + 12 * i)) % 9) + 1) * (-1) ** ((i + 3) ** 2 % 7)
            ex_cnt += 1
        while (z * t1 - t3 * s1 == 0):
            t3 = -((((i ** 3 + 7 * ex_cnt) % (15 + 12 * i)) % 7) + 1) * (-1) ** ((i + 2) ** 2 % 7)
            ex_cnt += 1
        line1 = ['a', s1, s2, z]
        line2 = [0, t1, t2, t3]
        line3 = [u1, 0, 'b', u2]
        if (swap_2_3_row):
            line2, line3 = line3, line2
        if (swap_2_3_col):
            line1[1], line1[2] = line1[2], line1[1]
            line2[1], line2[2] = line2[2], line2[1]
            line3[1], line3[2] = line3[2], line3[1]
        ans_ab = Fraction(u1 * t1 * s2 - s1 * t2 * u1, t1)
        ans_a = Fraction(z * u1 * t1 - t3 * s1 * u1, u2 * t1)
        ans_b = Fraction(ans_ab, ans_a)
    else:
        s1 = -((((i ** 4 + 3) % (7 + 11 * i)) % 7) + 1) * (-1) ** ((i + 5) ** 4 % 2)
        s2 = -((((i ** 4 + 5) % (7 + 11 * i)) % 5) + 1) * (-1) ** ((i + 5) ** 2 % 4)
        t1 = -((((i ** 3 + 2) % (7 + 12 * i)) % 6) + 1) * (-1) ** ((i + 5) ** 6 % 7)
        t2 = 1 * (-1) ** ((i + 5) ** 3 % 3)
        t3 = -((((i ** 3 + 1) % (7 + 13 * i)) % 5) + 1) * (-1) ** ((i + 5) ** 5 % 5 + 1)
        u1 = -((((i ** 5 + 2) % (7 + 10 * i)) % 8) + 1) * (-1) ** ((i + 4) ** 3 % 5)
        u2 = -((((i ** 6 + 3) % (11 + 12 * i)) % 9) + 1) * (-1) ** ((i + 7) ** 3 % 3)
        z = -((((i ** 2 + 5) % (13 + 12 * i)) % 7) + 1) * (-1) ** ((i + 4) ** 3 % 6)
        while (u2 * t1 - t2 * u1 == 0):
            t1 = -((((i ** 4 + 3 * ex_cnt) % (17 + 12 * i)) % 9) + 1) * (-1) ** ((i + 3) ** 2 % 7)
            ex_cnt += 1
        while (z * t2 - t3 * u2 == 0):
            t3 = -((((i ** 3 + 7 * ex_cnt) % (15 + 12 * i)) % 7) + 1) * (-1) ** ((i + 2) ** 2 % 7)
            ex_cnt += 1
        line1 = ['a', 0, s1, s2]
        line2 = [t1, t2, 0, t3]
        line3 = [u1, u2, 'b', z]
        if (swap_2_3_row):
            line2, line3 = line3, line2
        if (swap_2_3_col):
            line1[1], line1[2] = line1[2], line1[1]
            line2[1], line2[2] = line2[2], line2[1]
            line3[1], line3[2] = line3[2], line3[1]
        ans_ab = Fraction(t2 * u1 * s1 - t1 * u2 * s1, t2)
        ans_b = Fraction(t2 * z * s1 - t3 * s1 * u2, s2 * t2)
        ans_a = Fraction(ans_ab, ans_b)
    # готовим строки для вывода
    container = ['x', 'y', 'z']
    for j in range(3):
        if (type(line1[j]) != type(0)):
            str_1 = str_1 + ('+' if len(str_1) != 0 else "") + str(line1[j]) + container[j]
        else:
            if (line1[j] != 0):
                str_1 = str_1 + ('+' if line1[j] > 0 and len(str_1) != 0 else "") + ('-' if line1[j] == -1 else str(line1[j]) if line1[j] != 1 else "") + container[j]
    for j in range(3):
        if (type(line2[j]) != type(0)):
            str_2 = str_2 + ('+' if len(str_2) != 0 else "") + (str(line2[j])) + container[j]
        else:
            if (line2[j] != 0):
                str_2 = str_2 + ('+' if line2[j] > 0 and len(str_2) != 0 else "") + ('-' if line2[j] == -1 else str(line2[j]) if line2[j] != 1 else "") + container[j]
    for j in range(3):
        if (type(line3[j]) != type(0)):
            str_3 = str_3 + ('+' if len(str_3) != 0 else "") + str(line3[j]) + container[j]
        else:
            if (line3[j] != 0):
                str_3 = str_3 + ('+' if line3[j] > 0 and len(str_3) != 0 else "") + ('-' if line3[j] == -1 else str(line3[j]) if line3[j] != 1 else "") + container[j]
    b_part = [line1[3], line2[3], line3[3]]
    return str_1, str_2, str_3, ans_ab, ans_a, ans_b, b_part


def writeFourthTask(str_1, str_2, str_3, b_part, ans_ab, ans_a, ans_b):
    # код для внесения в task
    tasksFile.write("{\\noindent \\bf 4.} "
                    "Определите число решений следующей системы в зависимости от значений параметров $a$ и $b$:\n")

    tasksFile.write("\\[\\begin{cases}")
    tasksFile.write(str_1 + " &= " + str(b_part[0]) + ",\\\\")
    tasksFile.write(str_2 + " &= " + str(b_part[1]) + ",\\\\")
    tasksFile.write(str_3 + " &= " + str(b_part[2]) + ".")
    tasksFile.write("\\end{cases}"
                    "\\]\n"
                    "\\medskip\n")

    # код для внесения в ответы
    answersFile.write("{\\noindent 4.} Для данной СЛУ справедливы следующие утверждения: \\begin{itemize} \\item "
                      "При $ab \\ne " + latexFrac(ans_ab) + "$ СЛУ имеет единственное решение.\\\\\n\\item \n" +
                      "При $ab = " + latexFrac(ans_ab) + "$, $a = " + latexFrac(ans_a) + "$ и $b = " + latexFrac(ans_b) + "$ СЛУ имеет бесконечно много решений.\\\\\n\item \n" +
                      "При $ab = " + latexFrac(ans_ab) + "$, $a \\ne" + latexFrac(ans_a) + "$ и $b \\ne" + latexFrac(ans_b) + "$ СЛУ несовместна.\n"
                                                                                                                              "\\end{itemize}\n")


groups_size = 60
groups_number = 9
total_tasks = groups_number * groups_size
year = 2018

# Цикл, обходящий все группы. Запись в файл "tasks" и файл "answers" преамбулы ТеХ-файла
for index in range(1, groups_number + 1):
    # Создание файлов для записи условий и для записи ответов, подстановка номера группы в имена файлов
    tasksFile = open("18" + str(index) + "_tasks1.tex", 'w')
    answersFile = open("18" + str(index) + "_answers1.tex", 'w')

    latexHeader(tasksFile)
    latexHeader(answersFile)

    for i in range((index - 1) * groups_size + 1, index * groups_size + 1):
        group = 180 + int((i - 1) / groups_size) + 1
        variant = i - groups_size * int((i - 1) / groups_size)

        tasksHeader(tasksFile, group, variant)
        answersHeader(answersFile, group, variant)

        if len(inconsistentPrev) == 60:
            inconsistentPrev = []
        if len(infinitePrev) == 60:
            infinitePrev = []
        if len(equationsPrev) == 60:
            equationsPrev = []

        # Первое задание
        A1, B1, C1, D1, u1, v1 = generateFirstTask(i)
        writeFirstTask(A1, B1, C1, D1, u1, v1, tasksFile, answersFile, i)

        # Второе задание
        equations, infinite, inconsistent = generateSecondTask(i * 2 - i)
        writeSecondTask(equations, infinite, inconsistent, tasksFile, answersFile, i)

        # Третье задание
        A3, P3, X3 = generateThirdTask(i * 3 - i)
        writeThirdTask(A3, P3, X3, tasksFile, answersFile)

        # Четвертое задание
        str_1, str_2, str_3, ans_ab, ans_a, ans_b, b_part = generateFourthTask(i)
        writeFourthTask(str_1, str_2, str_3, b_part, ans_ab, ans_a, ans_b)

        tasksFile.write("\\newpage\n")
        answersFile.write("\\newpage\n")

    tasksFile.write("\\end{document}")
    answersFile.write("\\end{document}")
    tasksFile.close()
    answersFile.close()
