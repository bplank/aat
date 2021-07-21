##

#text:
python src/ml/classifier.py features/authorship/villani/text.csv --c-n-gram 2-3 --n-gram 1-2

#keystrokes basic
python src/ml/classifier.py features/authorship/villani/letters.csv

#keystrokes extended
python src/ml/classifier.py features/authorship/villani/linguistic.csv 

#keystrokes extended + text
python src/ml/classifier.py features/authorship/villani/linguistic+text.csv --c-n-gram 2-3 --n-gram 1-2

#keystrokes extended + embeds
python src/ml/classifier.py features/authorship/villani/linguistic+text.csv --embeds
