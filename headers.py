def latexHeader(file):
    file.write("\\documentclass{article}\n"
               "\\usepackage{latexsym,amsxtra,amscd,ifthen}\n"
               "\\usepackage{amsfonts}\n"
               "\\usepackage[bbgreekl]{mathbbol}\n"
               "\\usepackage{bbm}\n"
               "\\usepackage{verbatim}\n"
               "\\usepackage{amsmath}\n"
               "\\usepackage{amsthm}\n"
               "\\usepackage{amssymb}\n"
               "\\usepackage[inline, shortlabels]{enumitem}\n"
               "\\usepackage[russian]{babel}\n"
               "\\usepackage[utf8]{inputenc}\n"
               "\\usepackage[T2A]{fontenc}\n"
               "\\usepackage{nicefrac}\n"
               "\\pagestyle{plain}\n"
               "\\textwidth=19.0cm\n"
               "\\oddsidemargin=-1.3cm\n"
               "\\textheight=26cm\n"
               "\\topmargin=-3.0cm\n"
               "\\tolerance=500\n"
               "\\unitlength=1mm\n"
               "\\DeclareSymbolFontAlphabet{\\bbm}{bbold}\n"
               "\\DeclareSymbolFontAlphabet{\\mathbb}{AMSb}\n"
               "\\def\\R{{\\mathbb{R}}}\n"
               "\\begin{document}\n\n")


def tasksHeader(file, group, variant):
    group = str(group)
    variant = str(variant)
    file.write("\\begin{center}\n"
               "\\footnotesize\n"
               "\\noindent\\makebox[\\textwidth]{Линейная алгебра и геометрия \\hfill ФКН НИУ ВШЭ, 2018/2019 учебный год, 1-й курс ОП ПМИ, основной поток}\n"
               "\\end{center}\n"
               "\\begin{center}\n"
               "\\textbf{Индивидуальное домашнее задание 4}\n"
               "\\end{center}\n"
               "\\begin{center}\n"
               "{Группа БПМИ" + group + ". Вариант " + variant + "}\n\\end{center}\n\n")


def answersHeader(file, group, variant):
    group = str(group)
    variant = str(variant)
    file.write("\\begin{center}"
               "\\bf Ответы к индивидуальному домашнему заданию 4"
               "\\end{center}\n"
               "\\begin{center}"
               "{Группа БПМИ" + group + ". Вариант " + variant + "}\n\\end{center}\n\n")
