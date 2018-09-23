import numpy as np
import sympy as sp


def latexCdot(file):
    file.write("\\cdot")


def latexMatrix(file, A):
    file.write("\\begin{pmatrix}")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not isinstance(A[i, j], str):
                file.write(str(int(A[i, j])))
            else:
                file.write(A[i, j])
            if j != A.shape[1] - 1:
                file.write(" & ")
        if i != A.shape[0] - 1:
            file.write("\\\\")
    file.write("\\end{pmatrix}")


def generateSOLE():
    lowest_bound = -50
    highest_bound = 50
    first_basis = np.random.randint(lowest_bound, highest_bound, (4, 1))
    second_basis = np.random.randint(lowest_bound, highest_bound, (4, 1))
    while np.linalg.matrix_rank(np.matrix(np.column_stack((first_basis, second_basis)))) != 2:
        second_basis = np.random.randint(lowest_bound, highest_bound, (4, 1))
    column_b_infinite = (np.random.randint(int(lowest_bound / 10), int(highest_bound / 10)) * first_basis +
                         np.random.randint(int(lowest_bound / 10), int(highest_bound / 10)) * second_basis)
    third_vector = (np.random.randint(int(lowest_bound / 10), int(highest_bound / 10)) * first_basis +
                    np.random.randint(int(lowest_bound / 10), int(highest_bound / 10)) * second_basis)
    fourth_vector = (np.random.randint(int(lowest_bound / 10), int(highest_bound / 10)) * first_basis +
                     np.random.randint(int(lowest_bound / 10), int(highest_bound / 10)) * second_basis)
    column_b_inconsistent = np.random.randint(lowest_bound, highest_bound, (4, 1))
    while np.linalg.matrix_rank(np.matrix(np.column_stack((first_basis, second_basis, column_b_inconsistent)))) != 3:
        column_b_inconsistent = np.random.randint(lowest_bound, highest_bound, (4, 1))
    return np.matrix(np.column_stack((first_basis, second_basis, third_vector, fourth_vector))), column_b_infinite, column_b_inconsistent


def powerTask(i):
    M11 = np.matrix([[1, "a", 0], [0, "a", 0], [0, "a", 1]])
    M22 = np.matrix([["a", 1, 1], [0, "a ^ 2", 0], [0, 0, "a"]])
    M33 = np.matrix([["a", 0, 1], [0, "a ^ 2", 1], [0, 0, "a"]])
    M44 = np.matrix([["a", 0, 0], [1, "a", 1], [0, 0, "a"]])
    M55 = np.matrix([["a", 1, 0], [0, "a", 0], [0, 1, "a"]])
    M66 = np.matrix([[1, 0, 0], ["a", "a", "a"], [0, 0, 1]])

    a1 = (i + 1) % 4 + 2
    M1 = np.matrix([[1, a1, 0], [0, a1, 0], [0, a1, 1]])
    M2 = np.matrix([[a1, 1, 1], [0, a1 ** 2, 0], [0, 0, a1]])
    M3 = np.matrix([[a1, 0, 1], [0, a1 ** 2, 1], [0, 0, a1]])
    M4 = np.matrix([[a1, 0, 0], [1, a1, 1], [0, 0, a1]])
    M5 = np.matrix([[a1, 1, 0], [0, a1, 0], [0, 1, a1]])
    M6 = np.matrix([[1, 0, 0], [a1, a1, a1], [0, 0, 1]])
    SET = [M1, M2, M3, M4, M5, M6]

    SET1 = [M11, M22, M33, M44, M55, M66]

    R1 = np.matrix([[1, "\\frac{a(a^n - 1)}{a - 1}", 0],
                    [0, "a ^ n", 0],
                    [0, "\\frac{a(a^n - 1)}{a - 1}", 1]])

    R2 = np.matrix([["a ^ n", "t", "na ^ {(n - 1)}"],
                    [0, "a ^ {2n}", 0],
                    [0, 0, "a ^ n"]])

    R3 = np.matrix([["a ^ n", 0, "na ^ {(n - 1)}"],
                    [0, "a ^ {2n}", "t"],
                    [0, 0, "a ^ n"]])

    R4 = np.matrix([["a ^ n", 0, 0],
                    ["na ^ {(n - 1)}", "a ^ n", "na ^ {(n - 1)}"],
                    [0, 0, "a ^ n"]])

    R5 = np.matrix([["a ^ n", "na ^ {(n - 1)}", 0],
                    [0, "a ^ n", 0],
                    [0, "na ^ {(n - 1)}", "a ^ n"]])

    R6 = np.matrix([[1, 0, 0],
                    ["\\frac{a(a^n - 1)}{a - 1}", "a ^ n", "\\frac{a(a^n - 1)}{a - 1}"],
                    [0, 0, 1]])

    R = [R1, R2, R3, R4, R5, R6]
    RES = [SET[i % 6], SET1[i % 6], R[i % 6]]
    return RES


generateSOLE()

groups_size = 60
groups_number = 9
total_tasks = groups_number * groups_size
year = 2018

tasks_filename = ["" for i in range(groups_number + 2)]
answers_filename = ["" for i in range(groups_number + 2)]
# Цикл, обходящий все группы. Запись в файл "tasks" и файл "answers" преамбулы ТеХ-файла
for index in range(1, groups_number + 1):
    # Создание файлов для записи условий и для записи ответов, подстановка номера группы в имена файлов
    tasks_filename[index] = "18" + str(index) + "_tasks.tex"
    answers_filename[index] = "18" + str(index) + "_answers.tex"
    tasks_file = open(tasks_filename[index], 'w')
    answers_file = open(answers_filename[index], 'w')

    tasks_file.write("\\documentclass{article}\n"
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

    answers_file.write("\\documentclass{article}\n"
                       "\\usepackage{latexsym,amsxtra,amscd,ifthen}\n"
                       "\\usepackage{amsfonts}\n"
                       "\\usepackage{verbatim}\n"
                       "\\usepackage{amsmath}\n"
                       "\\usepackage{amsthm}\n"
                       "\\usepackage{amssymb}\n"
                       "\\usepackage[russian]{babel}\n"
                       "\\usepackage[utf8]{inputenc}\n"
                       "\\numberwithin{equation}{section}\n"
                       "\\pagestyle{plain}\n"
                       "\\tolerance=500\n"
                       "\\unitlength=1mm\n"
                       "\\textwidth=16cm\n"
                       "\\textheight=770pt\n"
                       "\\oddsidemargin=-8mm\n"
                       "\\topmargin -32mm\n"
                       "\\def\\R{{\\mathbb{R}}}\n"
                       "\\begin{document}\n")

    for i in range((index - 1) * groups_size + 1, index * groups_size + 1):
        # Первое задание
        EQ, b_infinite, b_inconsistent = generateSOLE()
        EQ_with_infinite = np.c_[EQ, b_infinite]
        EQ_with_inconsistent = np.c_[EQ, b_inconsistent]
        row_reduced_infinite = np.matrix(sp.Matrix(EQ_with_infinite).rref()[0])
        row_reduced_inconsistent = np.matrix(sp.Matrix(EQ_with_inconsistent).rref()[0])

        C = [str(int(EQ[0, 0])) + "x_1" +
             ("+" if EQ[0, 1] > 0 else "") + str(int(EQ[0, 1])) + "x_2" +
             ("+" if EQ[0, 2] > 0 else "") + str(int(EQ[0, 2])) + "x_3" +
             ("+" if EQ[0, 3] > 0 else "") + str(int(EQ[0, 3])) + "x_4",

             str(int(EQ[1, 0])) + "x_1" +
             ("+" if EQ[1, 1] > 0 else "") + str(int(EQ[1, 1])) + "x_2" +
             ("+" if EQ[1, 2] > 0 else "") + str(int(EQ[1, 2])) + "x_3" +
             ("+" if EQ[1, 3] > 0 else "") + str(int(EQ[1, 3])) + "x_4",

             str(int(EQ[2, 0])) + "x_1" +
             ("+" if EQ[2, 1] > 0 else "") + str(int(EQ[2, 1])) + "x_2" +
             ("+" if EQ[2, 2] > 0 else "") + str(int(EQ[2, 2])) + "x_3" +
             ("+" if EQ[2, 3] > 0 else "") + str(int(EQ[2, 3])) + "x_4",

             str(int(EQ[3, 0])) + "x_1" +
             ("+" if EQ[3, 1] > 0 else "") + str(int(EQ[3, 1])) + "x_2" +
             ("+" if EQ[3, 2] > 0 else "") + str(int(EQ[3, 2])) + "x_3" +
             ("+" if EQ[3, 3] > 0 else "") + str(int(EQ[3, 3])) + "x_4"]

        # Второе задание
        lowest_bound = 1
        highest_bound = 9

        B2 = np.matrix(np.random.randint(lowest_bound, highest_bound, (2, 2)))
        C2 = np.matrix(np.random.randint(lowest_bound, highest_bound, (2, 2)))
        D2 = np.matrix(np.random.randint(lowest_bound, highest_bound, (2, 2)))
        Ans_T = []

        A2 = C2 + D2

        if ((i + 4) ** 3) % 3 == 0:
            Ans_T = (A2.transpose() * C2 * B2 -
                     (B2 * A2).transpose() * B2 +
                     A2.transpose() * D2 * B2 -
                     C2.transpose() * A2 * A2.transpose() +
                     A2.transpose() * (A2 * B2).transpose() -
                     D2.transpose() * A2 * A2.transpose())

        elif ((i + 4) ** 3) % 3 == 1:
            Ans_T = (C2.transpose() * B2 * A2.transpose() -
                     A2.transpose() * B2 * B2 +
                     D2.transpose() * B2 * A2.transpose() -
                     A2.transpose() * A2 * A2.transpose() +
                     A2.transpose() * C2 * B2 +
                     A2.transpose() * C2 * D2)

        else:
            Ans_T = (D2 * B2 * B2.transpose() +
                     C2 * (C2 * B2).transpose() -
                     A2 * B2.transpose() * B2.transpose() +
                     D2 * (C2 * B2).transpose() +
                     A2 * (D2 * B2).transpose() +
                     C2 * B2 * B2)

        # Третье задание
        A_power_n = powerTask(i)

        # Запись в файл "tasks" шапки документа, номера группы и номера варианта
        group = 180 + int((i - 1) / groups_size) + 1
        variant = i - groups_size * int((i - 1) / groups_size)
        tasks_file.write("\\begin{center}\n"
                         "\\footnotesize\n"
                         "\\noindent\\makebox[\\textwidth]{Линейная алгебра и геометрия \\hfill ФКН НИУ ВШЭ, 2018/2019 учебный год, 1-й курс ОП ПМИ, основной поток}\n"
                         "\\end{center}\n"
                         "\\begin{center}\n"
                         "\\textbf{Индивидуальное домашнее задание 1}\n"
                         "\\end{center}\n"
                         "\\begin{center}\n"
                         "{Группа БПМИ" + str(group) + ". Вариант " + str(variant) + "}\n"
                                                                                     "\\end{center}\n")

        random_value = int(2 * ((2 ** (1 / 2) * i + 0.2) - int(2 ** (1 / 2) * i + 0.2)))

        # Запись в файл "tasks" условия задачи 1
        tasks_file.write("{\\noindent \\bf 1.1} "
                         "Решите приведённую ниже систему линейных уравнений методом Гаусса. "
                         "Если система совместна, то выпишите её общее решение и укажите одно частное решение.")
        b_column = ()
        if random_value == 1:  # Запись несовместной системы сначала
            b_column = (b_inconsistent, b_infinite)
        else:  # Запись системы с бесконечным количеством решений сначала
            b_column = (b_infinite, b_inconsistent)

        tasks_file.write("\\[\\begin{cases}")
        tasks_file.write(C[0] + " &= " + str(int(b_column[0][0])) + "\\\\")
        tasks_file.write(C[1] + " &= " + str(int(b_column[0][1])) + "\\\\")
        tasks_file.write(C[2] + " &= " + str(int(b_column[0][2])) + "\\\\")
        tasks_file.write(C[3] + " &= " + str(int(b_column[0][3])))
        tasks_file.write("\\end{cases}"
                         "\\]\n"
                         "\\medskip\n")

        tasks_file.write("{\\noindent \\bf 1.2} "
                         "Тот же вопрос для этой системы.")

        tasks_file.write("\\[\\begin{cases}")
        tasks_file.write(C[0] + " &= " + str(int(b_column[1][0])) + "\\\\")
        tasks_file.write(C[1] + " &= " + str(int(b_column[1][1])) + "\\\\")
        tasks_file.write(C[2] + " &= " + str(int(b_column[1][2])) + "\\\\")
        tasks_file.write(C[3] + " &= " + str(int(b_column[1][3])))
        tasks_file.write("\\end{cases}"
                         "\\]\n"
                         "\\medskip\n")

        # Запись в файл "tasks" условия задачи 2
        tasks_file.write("{\\noindent \\bf2.} "
                         "Вычислите: ")
        if (i + 4) ** 3 % 3 == 0:
            tasks_file.write("\\[")
            latexMatrix(tasks_file, A2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, C2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)

            tasks_file.write("-")

            tasks_file.write("\\Biggl[")
            latexMatrix(tasks_file, B2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2)
            tasks_file.write("\\Biggl]^{T}")
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)

            tasks_file.write("+")

            latexMatrix(tasks_file, A2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, D2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)

            tasks_file.write("-")
            tasks_file.write("\\]")
            tasks_file.write("\\[")
            tasks_file.write("-")

            latexMatrix(tasks_file, C2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2.transpose())

            tasks_file.write("+")

            latexMatrix(tasks_file, A2.transpose())
            latexCdot(tasks_file)
            tasks_file.write("\\Biggl[")
            latexMatrix(tasks_file, A2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)
            tasks_file.write("\\Biggl]^{T}")

            tasks_file.write("-")

            latexMatrix(tasks_file, D2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2.transpose())

            tasks_file.write("\\]")

        elif (i + 4) ** 3 % 3 == 1:
            tasks_file.write("\\[")

            latexMatrix(tasks_file, C2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2.transpose())

            tasks_file.write("-")

            latexMatrix(tasks_file, A2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2)
            tasks_file.write("^{2}")

            tasks_file.write("+")

            latexMatrix(tasks_file, D2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2.transpose())

            tasks_file.write("-")
            tasks_file.write("\\]")
            tasks_file.write("\\[")
            tasks_file.write("-")

            latexMatrix(tasks_file, A2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, A2.transpose())

            tasks_file.write("+")

            latexMatrix(tasks_file, A2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, C2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)

            tasks_file.write("+")

            latexMatrix(tasks_file, A2.transpose())
            latexCdot(tasks_file)
            latexMatrix(tasks_file, C2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, D2)

            tasks_file.write("\\]")

        else:
            tasks_file.write("\\[")

            latexMatrix(tasks_file, D2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2.transpose())

            tasks_file.write("+")

            latexMatrix(tasks_file, C2)
            latexCdot(tasks_file)
            tasks_file.write("\\Biggl[")
            latexMatrix(tasks_file, C2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)
            tasks_file.write("\\Biggl]^{T}")

            tasks_file.write("-")

            latexMatrix(tasks_file, A2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2.transpose())
            tasks_file.write("^{2}")

            tasks_file.write("+")
            tasks_file.write("\\]")
            tasks_file.write("\\[")
            tasks_file.write("+")

            latexMatrix(tasks_file, D2)
            latexCdot(tasks_file)
            tasks_file.write("\\Biggl[")
            latexMatrix(tasks_file, C2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)
            tasks_file.write("\\Biggl]^{T}")

            tasks_file.write("+")

            latexMatrix(tasks_file, A2)
            latexCdot(tasks_file)
            tasks_file.write("\\Biggl[")
            latexMatrix(tasks_file, D2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)
            tasks_file.write("\\Biggl]^{T}")

            tasks_file.write("+")

            latexMatrix(tasks_file, C2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)
            latexCdot(tasks_file)
            latexMatrix(tasks_file, B2)

            tasks_file.write("\\]")
        tasks_file.write("\n\\medskip\\\\\n")

        # Запись в файл "tasks" условия задачи 3
        tasks_file.write("{\\noindent \\bf 3.} Вычислите $A^n$, где \\[A=")
        latexMatrix(tasks_file, np.matrix(A_power_n[0]))
        tasks_file.write("\\]\n")

        tasks_file.write("\\newpage")

        # Запись в файл "answers" заголовка
        answers_file.write("\\begin{center}"
                           "\\bf Ответы к индивидуальному домашнему заданию 1"
                           "\\end{center}"
                           "\\begin{center}"
                           "{Группа БПМИ" + str(group) + ". Вариант " + str(variant) + "}\n\\end{center}\n")

        # Запись в файл "answers" ответа к задаче 1
        if random_value == 1:
            answers_file.write("\\[\\]"
                               "1.1. Должна получиться несовместная система со следующей матрицей: $$")
            latexMatrix(answers_file, row_reduced_inconsistent)
            answers_file.write("$$\n")

            answers_file.write("\\[\\]"
                               "1.2. Должна получиться совместная система с бесконечным количеством решений и следующей матрицей: $$")
            latexMatrix(answers_file, row_reduced_infinite)
            answers_file.write("$$\n")
            answers_file.write("\\[\\] Общее решение данной СЛУ:"
                               "\\begin{itemize}"
                               "\\item"
                               "$x_1 = " + str(0 - int(row_reduced_infinite[0, 2])) + "\\cdot x_3" +
                               ("+" if row_reduced_infinite[0, 3] < 0 else "") + str(0 - int(row_reduced_infinite[0, 3])) + "\\cdot x_4" +
                               ("+" if row_reduced_infinite[0, 4] >= 0 else "") + str(int(row_reduced_infinite[0, 4])) +
                               "$\\item"
                               "$x_2 = " + str(0 - int(row_reduced_infinite[1, 2])) + "\\cdot x_3" +
                               ("+" if row_reduced_infinite[1, 3] < 0 else "") + str(0 - int(row_reduced_infinite[1, 3])) + "\\cdot x_4" +
                               ("+" if row_reduced_infinite[1, 4] >= 0 else "") + str(int(row_reduced_infinite[1, 4])) +
                               "$\\item $x_3 \\text{ любое }$ \\item $x_4 \\text{ любое }$\\end{itemize}\n")

        else:
            answers_file.write("\\[\\]"
                               "1.1. Должна получиться совместная система с бесконечным количеством решений и следующей матрицей: $$")
            latexMatrix(answers_file, row_reduced_infinite)
            answers_file.write("$$\n")
            answers_file.write("\\[\\] Общее решение данной СЛУ:"
                               "\\begin{itemize}"
                               "\\item"
                               "$x_1 = " + str(0 - int(row_reduced_infinite[0, 2])) + "\\cdot x_3" +
                               ("+" if row_reduced_infinite[0, 3] < 0 else "") + str(0 - int(row_reduced_infinite[0, 3])) + "\\cdot x_4" +
                               ("+" if row_reduced_infinite[0, 4] >= 0 else "") + str(int(row_reduced_infinite[0, 4])) +
                               "$\\item"
                               "$x_2 = " + str(0 - int(row_reduced_infinite[1, 2])) + "\\cdot x_3" +
                               ("+" if row_reduced_infinite[1, 3] < 0 else "") + str(0 - int(row_reduced_infinite[1, 3])) + "\\cdot x_4" +
                               ("+" if row_reduced_infinite[1, 4] >= 0 else "") + str(int(row_reduced_infinite[1, 4])) +
                               "$\\item $x_3 \\text{ любое }$ \\item $x_4 \\text{ любое }$\\end{itemize}\n")

            answers_file.write("\\[\\]"
                               "1.2. Должна получиться несовместная система со следующей матрицей: $$")
            latexMatrix(answers_file, row_reduced_inconsistent)
            answers_file.write("$$\n")


        # Запись в файл "answers" ответа к задаче 2
        answers_file.write("\\[\\]2. $")
        latexMatrix(answers_file, Ans_T)
        answers_file.write("$\n")

        # Запись в файл "answers" ответа к задаче 3
        answers_file.write("\\[\\]3. \\[ A=")
        latexMatrix(answers_file, np.matrix(A_power_n[1]))
        answers_file.write("\\]\n \\[ A^n=")
        latexMatrix(answers_file, np.matrix(A_power_n[2]))
        answers_file.write("\\]\n")
        if i % 6 == 1 or i % 6 == 2:
            answers_file.write("где $t={\\sum \\limits_{k=n-1}^{2n-2}{a^k}}$\n")

        answers_file.write("\n\\newpage\n")

    tasks_file.write("\\end{document}")
    answers_file.write("\\end{document}")

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