import numpy as np
import sympy as sp


def latexHeader(file):
    file.write("\\documentclass{article}\n"
               "\\usepackage{latexsym,amsxtra,amscd,ifthen}\n"
               "\\usepackage{amsfonts}\n"
               "\\usepackage{verbatim}\n"
               "\\usepackage{amsmath}\n"
               "\\usepackage{amsthm}\n"
               "\\usepackage{amssymb}\n"
               "\\usepackage[russian]{babel}\n"
               "\\usepackage[utf8]{inputenc}\n"
               "\\usepackage[T2A]{fontenc}\n"
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
            file.write(str(A[i, j]))
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


def generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, rows, columns):
    matrix = np.random.randint(lowest_bound, highest_bound, (rows, columns), int)
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            while matrix[row][column] == 0:
                matrix[row][column] = np.random.randint(lowest_bound, highest_bound)
    return np.matrix(matrix)


def generateFirstTask(lowest_bound, highest_bound, u_lowest_bound, u_highest_bound, v_lowest_bound, v_highest_bound):
    A = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 3)
    B = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 3)
    C = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 2)
    D = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 2)
    while np.array_equal(C * D, D * C):
        C = generateRandomMatrixWithNoZeros(lowest_bound, highest_bound, 2, 2)
    u = 0
    while u == -1 or u == 0 or u == 1:
        u = np.random.randint(u_lowest_bound, u_highest_bound)
    v = 0
    while v == -1 or v == 0 or v == 1:
        v = np.random.randint(v_lowest_bound, v_highest_bound)
    return A, B, C, D, u, v


def writeFirstTask(A, B, C, D, u, v, tasksFile, answers):
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

    swipe_CD1 = np.random.randint(0, 2)
    swipe_C1 = np.random.randint(0, 2)
    swipe_AB12 = np.random.randint(0, 2)

    tasksFile.write(str(u))
    # Начало первого слагаемого
    first_term = 1
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

    swipe_AB2 = np.random.randint(0, 2)
    swipe_sign2 = np.random.randint(0, 2)
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

    # if v > 0:
    #     tasksFile.write("+ ")
    # tasksFile.write("\\]")
    # tasksFile.write("\\[")

    # Начало третьего слагаемого
    third_term = v * C * C
    if v > 0:
        tasksFile.write("+ ")
    tasksFile.write(str(v))
    tasksFile.write("C^2")
    # Конец третьего слагаемого

    # Начало четвертого слагаемого
    fourth_term = 1
    swipe_sign4 = np.random.randint(0, 2)
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


def generateSecondTask(numbers_lowest_bound, numbers_highest_bound, coefficient_lowest_bound, coefficient_highest_bound):
    first_basis = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))
    second_basis = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))

    while np.linalg.matrix_rank(np.matrix(np.column_stack((first_basis, second_basis)))) != 2:
        second_basis = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))

    third_vector = (np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * first_basis +
                    np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * second_basis)
    fourth_vector = (np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * first_basis +
                     np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * second_basis)

    column_b_infinite = (np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * first_basis +
                         np.random.randint(coefficient_lowest_bound, coefficient_highest_bound) * second_basis)

    column_b_inconsistent = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))
    while np.linalg.matrix_rank(np.matrix(np.column_stack((first_basis, second_basis, column_b_inconsistent)))) != 3:
        column_b_inconsistent = np.random.randint(numbers_lowest_bound, numbers_highest_bound, (4, 1))

    return np.matrix(np.column_stack((first_basis, second_basis, third_vector, fourth_vector))), column_b_infinite, column_b_inconsistent


def writeSOLE(equations, b, file, sign):
    file.write("$\\begin{cases}")
    for row in range(equations.shape[0]):
        for column in range(equations.shape[1]):
            if equations[row, column] == 0:
                continue
            if column != 0:
                file.write("+" if equations[row, column] > 0 else "")
            if equations[row, column] != 1 and equations[row, column] != 1:
                file.write(str(equations[row, column]))
            elif equations[row, column] == -1:
                file.write("-")
            file.write("x_{" + str(column + 1) + "}")
        file.write(" &= " + str(int(b[row][0])))
        if row < len(equations) - 1:
            file.write(", \\\\")
        else:
            file.write(sign)
    file.write("\\end{cases}$ \n \\medskip\n")


def writeSecondTask(equations, infinite, inconsistent, tasksFile, answersFile):
    inconsistentFirst = np.random.randint(0, 2)
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


groups_size = 60
groups_number = 9
total_tasks = groups_number * groups_size
year = 2018

# Цикл, обходящий все группы. Запись в файл "tasks" и файл "answers" преамбулы ТеХ-файла
for index in range(1, groups_number + 1):
    # Создание файлов для записи условий и для записи ответов, подстановка номера группы в имена файлов
    tasksFile = open("18" + str(index) + "_tasks.tex", 'w')
    answersFile = open("18" + str(index) + "_answers.tex", 'w')

    latexHeader(tasksFile)
    latexHeader(answersFile)

    for i in range((index - 1) * groups_size + 1, index * groups_size + 1):
        group = 180 + int((i - 1) / groups_size) + 1
        variant = i - groups_size * int((i - 1) / groups_size)

        tasksHeader(tasksFile, group, variant)
        answersHeader(answersFile, group, variant)

        # Первое задание
        A1, B1, C1, D1, u1, v1 = generateFirstTask(-5, 6, -5, 6, -5, 6)
        writeFirstTask(A1, B1, C1, D1, u1, v1, tasksFile, answersFile)

        # Второе задание
        equations, infinite, inconsistent = generateSecondTask(-50, 50, -5, 5)
        writeSecondTask(equations, infinite, inconsistent, tasksFile, answersFile)

        tasksFile.write("\\newpage\n")
        answersFile.write("\\newpage\n")

    tasksFile.write("\\end{document}")
    answersFile.write("\\end{document}")
    tasksFile.close()
    answersFile.close()

# B2[0, 0] = ((i ** 3 + 4) % 7 + i) % 5 + 2
# B2[0, 1] = ((i ** 3 + 1) % 7 + i) % 8 + 1
# B2[1, 0] = ((i ** 3 + 1) % 7 + i) % 7 + 1
# B2[1, 1] = ((i ** 3 + 1) % 7 + i) % 5 + 1
#
# C2[0, 0] = ((i ** 3 + 4) % 7 + i) % 5 + 2
# C2[0, 1] = ((i ** 3 + 1) % 7 + i) % 8 + 3
# C2[1, 0] = ((i ** 3 + 1) % 7 + i) % 7 + 1
# C2[1, 1] = ((i ** 3 + 1) % 7 + i) % 5 + 2
#
# D2[0, 0] = ((i ** 3 + 4) % 5 + i) % 5 + 1
# D2[0, 1] = ((i ** 3 + 1) % 7 + i) % 8 + 4
# D2[1, 0] = ((i ** 3 + 1) % 5 + i) % 7 + 1
# D2[1, 1] = ((i ** 3 + 1) % 7 + i) % 3 + 2

# def basisMat(i):
#     A = np.matrix(np.zeros((4, 4)))
#
#     A[0, 0] = (((i ** 3 + 10) % 7 + i) % 6 + 1) * (-1) ** (((i + 1) ** 2) % 7)
#     A[0, 1] = (((i ** 3 + 12) % 7 + i) % 10 + 1) * (-1) ** (((i + 5) ** 2) % 7)
#     A[0, 2] = (((i ** 3 + 13) % 7 + i) % 6 + 1) * (-1) ** (((i + 2) ** 2) % 7)
#     A[0, 3] = (((i ** 3 + 3) % 7 + i) % 10 + 1) * (-1) ** (((i + 7) ** 2) % 7)
#     A[1, 0] = (((i ** 3 + 15) % 7 + i) % 7 + 1) * (-1) ** (((i + 3) ** 2) % 7)
#     A[1, 1] = (((i ** 3 + 17) % 7 + i) % 8 + 1) * (-1) ** (((i + 4) ** 2) % 7)
#     A[1, 2] = (((i ** 3 + 18) % 7 + i) % 6 + 1) * (-1) ** (((i + 6) ** 2) % 7)
#     A[1, 3] = (((i ** 3 + 5) % 7 + i) % 7 + 1) * (-1) ** (((i + 10) ** 2) % 7)
#     A[2, 0] = (((i ** 3 + 19) % 7 + i) % 10 + 1) * (-1) ** (((i + 7) ** 2) % 7)
#     A[2, 1] = (((i ** 3 + 21) % 7 + i) % 7 + 1) * (-1) ** (((i + 8) ** 2) % 7)
#     A[2, 2] = (((i ** 3 + 22) % 7 + i) % 10 + 1) * (-1) ** (((i + 5) ** 2) % 7)
#     A[2, 3] = (((i ** 3 + 6) % 7 + i) % 6 + 1) * (-1) ** (((i + 6) ** 2) % 7)
#     A[3, 0] = (((i ** 3 + 2) % 7 + i) % 6 + 1) * (-1) ** (((i + 6) ** 2) % 7)
#     A[3, 1] = (((i ** 3 + 4) % 7 + i) % 7 + 1) * (-1) ** (((i + 9) ** 2) % 7)
#     A[3, 2] = (((i ** 3 + 5) % 7 + i) % 10 + 1) * (-1) ** (((i + 5) ** 2) % 7)
#     A[3, 3] = (((i ** 3 + 6) % 7 + i) % 7 + 1) * (-1) ** (((i + 3) ** 2) % 7)
#
#     if np.linalg.det(A) != 0:
#         return A
#     else:
#         return basisMat(i + 15)
#
#
# def simpleBasis(i):
#     A = np.matrix(np.zeros((4, 4)))
#
#     A[0, 0] = (((i ** 3 + 10) % 7 + i) % 3 + 1) * (-1) ** (((i + 1) ** 2) % 7)
#     A[0, 1] = (((i ** 3 + 12) % 7 + i) % 4 + 1) * (-1) ** (((i + 5) ** 2) % 7)
#     A[0, 2] = (((i ** 3 + 13) % 7 + i) % 3 + 1) * (-1) ** (((i + 2) ** 2) % 7)
#     A[0, 3] = (((i ** 3 + 3) % 7 + i) % 3 + 1) * (-1) ** (((i + 7) ** 2) % 7)
#     A[1, 0] = (((i ** 3 + 15) % 7 + i) % 3 + 1) * (-1) ** (((i + 3) ** 2) % 7)
#     A[1, 1] = (((i ** 3 + 17) % 7 + i) % 4 + 1) * (-1) ** (((i + 4) ** 2) % 7)
#     A[1, 2] = (((i ** 3 + 18) % 7 + i) % 4 + 1) * (-1) ** (((i + 6) ** 2) % 7)
#     A[1, 3] = (((i ** 3 + 5) % 7 + i) % 3 + 1) * (-1) ** (((i + 10) ** 2) % 7)
#     A[2, 0] = (((i ** 3 + 19) % 7 + i) % 3 + 1) * (-1) ** (((i + 7) ** 2) % 7)
#     A[2, 1] = (((i ** 3 + 21) % 7 + i) % 4 + 1) * (-1) ** (((i + 8) ** 2) % 7)
#     A[2, 2] = (((i ** 3 + 22) % 7 + i) % 3 + 1) * (-1) ** (((i + 5) ** 2) % 7)
#     A[2, 3] = (((i ** 3 + 6) % 7 + i) % 4 + 1) * (-1) ** (((i + 6) ** 2) % 7)
#     A[3, 0] = (((i ** 3 + 2) % 7 + i) % 3 + 1) * (-1) ** (((i + 6) ** 2) % 7)
#     A[3, 1] = (((i ** 3 + 4) % 7 + i) % 4 + 1) * (-1) ** (((i + 9) ** 2) % 7)
#     A[3, 2] = (((i ** 3 + 5) % 7 + i) % 4 + 1) * (-1) ** (((i + 5) ** 2) % 7)
#     A[3, 3] = (((i ** 3 + 6) % 7 + i) % 3 + 1) * (-1) ** (((i + 3) ** 2) % 7)
#
#     if np.linalg.det(A) != 0:
#         return A
#     else:
#         return simpleBasis(i + 15)
#
#
# def powerTask(i):
#     M11 = np.matrix([[1, "a", 0], [0, "a", 0], [0, "a", 1]])
#     M22 = np.matrix([["a", 1, 1], [0, "a ^ 2", 0], [0, 0, "a"]])
#     M33 = np.matrix([["a", 0, 1], [0, "a ^ 2", 1], [0, 0, "a"]])
#     M44 = np.matrix([["a", 0, 0], [1, "a", 1], [0, 0, "a"]])
#     M55 = np.matrix([["a", 1, 0], [0, "a", 0], [0, 1, "a"]])
#     M66 = np.matrix([[1, 0, 0], ["a", "a", "a"], [0, 0, 1]])
#
#     a1 = (i + 1) % 4 + 2
#     M1 = np.matrix([[1, a1, 0], [0, a1, 0], [0, a1, 1]])
#     M2 = np.matrix([[a1, 1, 1], [0, a1 ** 2, 0], [0, 0, a1]])
#     M3 = np.matrix([[a1, 0, 1], [0, a1 ** 2, 1], [0, 0, a1]])
#     M4 = np.matrix([[a1, 0, 0], [1, a1, 1], [0, 0, a1]])
#     M5 = np.matrix([[a1, 1, 0], [0, a1, 0], [0, 1, a1]])
#     M6 = np.matrix([[1, 0, 0], [a1, a1, a1], [0, 0, 1]])
#     SET = [M1, M2, M3, M4, M5, M6]
#
#     SET1 = [M11, M22, M33, M44, M55, M66]
#
#     R1 = np.matrix([[1, "\\frac{a(a^n - 1)}{a - 1}", 0],
#                     [0, "a ^ n", 0],
#                     [0, "\\frac{a(a^n - 1)}{a - 1}", 1]])
#
#     R2 = np.matrix([["a ^ n", "t", "na ^ {(n - 1)}"],
#                     [0, "a ^ {2n}", 0],
#                     [0, 0, "a ^ n"]])
#
#     R3 = np.matrix([["a ^ n", 0, "na ^ {(n - 1)}"],
#                     [0, "a ^ {2n}", "t"],
#                     [0, 0, "a ^ n"]])
#
#     R4 = np.matrix([["a ^ n", 0, 0],
#                     ["na ^ {(n - 1)}", "a ^ n", "na ^ {(n - 1)}"],
#                     [0, 0, "a ^ n"]])
#
#     R5 = np.matrix([["a ^ n", "na ^ {(n - 1)}", 0],
#                     [0, "a ^ n", 0],
#                     [0, "na ^ {(n - 1)}", "a ^ n"]])
#
#     R6 = np.matrix([[1, 0, 0],
#                     ["\\frac{a(a^n - 1)}{a - 1}", "a ^ n", "\\frac{a(a^n - 1)}{a - 1}"],
#                     [0, 0, 1]])
#
#     R = [R1, R2, R3, R4, R5, R6]
#     RES = [SET[i % 6], SET1[i % 6], R[i % 6]]
#     return RES
# # Второе задание
# lowest_bound = 1
# highest_bound = 9
#
# B2 = np.matrix(np.random.randint(lowest_bound, highest_bound, (2, 2)))
# C2 = np.matrix(np.random.randint(lowest_bound, highest_bound, (2, 2)))
# D2 = np.matrix(np.random.randint(lowest_bound, highest_bound, (2, 2)))
# Ans_T = []
#
# A2 = C2 + D2
#
# if ((i + 4) ** 3) % 3 == 0:
#     Ans_T = (A2.transpose() * C2 * B2 -
#              (B2 * A2).transpose() * B2 +
#              A2.transpose() * D2 * B2 -
#              C2.transpose() * A2 * A2.transpose() +
#              A2.transpose() * (A2 * B2).transpose() -
#              D2.transpose() * A2 * A2.transpose())
#
# elif ((i + 4) ** 3) % 3 == 1:
#     Ans_T = (C2.transpose() * B2 * A2.transpose() -
#              A2.transpose() * B2 * B2 +
#              D2.transpose() * B2 * A2.transpose() -
#              A2.transpose() * A2 * A2.transpose() +
#              A2.transpose() * C2 * B2 +
#              A2.transpose() * C2 * D2)
#
# else:
#     Ans_T = (D2 * B2 * B2.transpose() +
#              C2 * (C2 * B2).transpose() -
#              A2 * B2.transpose() * B2.transpose() +
#              D2 * (C2 * B2).transpose() +
#              A2 * (D2 * B2).transpose() +
#              C2 * B2 * B2)
#
# # Третье задание
# A_power_n = powerTask(i)
# # Запись в файл "tasks" условия задачи 2
# tasksFile.write("{\\noindent \\bf2.} "
#                  "Вычислите: ")
# if (i + 4) ** 3 % 3 == 0:
#     tasksFile.write("\\[")
#     latexMatrixProduct(tasksFile, [A2.transpose(), C2, B2])
#     tasksFile.write("-")
#
#     tasksFile.write("\\Biggl[")
#     latexMatrixProduct(tasksFile, [B2, A2])
#     tasksFile.write("\\Biggl]^{T}")
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#
#     tasksFile.write("+")
#
#     latexMatrixProduct(tasksFile, [A2.transpose(), D2, B2])
#
#     tasksFile.write("-")
#     tasksFile.write("\\]")
#     tasksFile.write("\\[")
#     tasksFile.write("-")
#
#     latexMatrix(tasksFile, C2.transpose())
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2.transpose())
#
#     tasksFile.write("+")
#
#     latexMatrix(tasksFile, A2.transpose())
#     latexCdot(tasksFile)
#     tasksFile.write("\\Biggl[")
#     latexMatrix(tasksFile, A2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#     tasksFile.write("\\Biggl]^{T}")
#
#     tasksFile.write("-")
#
#     latexMatrix(tasksFile, D2.transpose())
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2.transpose())
#
#     tasksFile.write("\\]")
#
# elif (i + 4) ** 3 % 3 == 1:
#     tasksFile.write("\\[")
#
#     latexMatrix(tasksFile, C2.transpose())
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2.transpose())
#
#     tasksFile.write("-")
#
#     latexMatrix(tasksFile, A2.transpose())
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2)
#     tasksFile.write("^{2}")
#
#     tasksFile.write("+")
#
#     latexMatrix(tasksFile, D2.transpose())
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2.transpose())
#
#     tasksFile.write("-")
#     tasksFile.write("\\]")
#     tasksFile.write("\\[")
#     tasksFile.write("-")
#
#     latexMatrix(tasksFile, A2.transpose())
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, A2.transpose())
#
#     tasksFile.write("+")
#
#     latexMatrix(tasksFile, A2.transpose())
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, C2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#
#     tasksFile.write("+")
#
#     latexMatrix(tasksFile, A2.transpose())
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, C2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, D2)
#
#     tasksFile.write("\\]")
#
# else:
#     tasksFile.write("\\[")
#
#     latexMatrix(tasksFile, D2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2.transpose())
#
#     tasksFile.write("+")
#
#     latexMatrix(tasksFile, C2)
#     latexCdot(tasksFile)
#     tasksFile.write("\\Biggl[")
#     latexMatrix(tasksFile, C2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#     tasksFile.write("\\Biggl]^{T}")
#
#     tasksFile.write("-")
#
#     latexMatrix(tasksFile, A2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2.transpose())
#     tasksFile.write("^{2}")
#
#     tasksFile.write("+")
#     tasksFile.write("\\]")
#     tasksFile.write("\\[")
#     tasksFile.write("+")
#
#     latexMatrix(tasksFile, D2)
#     latexCdot(tasksFile)
#     tasksFile.write("\\Biggl[")
#     latexMatrix(tasksFile, C2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#     tasksFile.write("\\Biggl]^{T}")
#
#     tasksFile.write("+")
#
#     latexMatrix(tasksFile, A2)
#     latexCdot(tasksFile)
#     tasksFile.write("\\Biggl[")
#     latexMatrix(tasksFile, D2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#     tasksFile.write("\\Biggl]^{T}")
#
#     tasksFile.write("+")
#
#     latexMatrix(tasksFile, C2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#     latexCdot(tasksFile)
#     latexMatrix(tasksFile, B2)
#
#     tasksFile.write("\\]")
# tasksFile.write("\n\\medskip\\\\\n")
#
# # Запись в файл "tasks" условия задачи 3
# tasksFile.write("{\\noindent \\bf 3.} Вычислите $A^n$, где \\[A=")
# latexMatrix(tasksFile, np.matrix(A_power_n[0]))
# tasksFile.write("\\]\n")
#