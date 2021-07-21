
# Text
python src/ml/classifier.py --n-gram 1 --c-n-gram 2-3 features/profiling/villani/text.csv  --meta features/profiling/villani/label_map_users2gender.txt
# keystrokes basic
python src/ml/classifier.py features/profiling/villani/letters.csv  --meta features/profiling/villani/label_map_users2gender.txt
# keystrokes ext
python src/ml/classifier.py features/profiling/villani/linguistic.csv  --meta features/profiling/villani/label_map_users2gender.txt
# keystrokes ext + text
python src/ml/classifier.py features/profiling/villani/linguistic+text.csv  --meta features/profiling/villani/label_map_users2gender.txt --n-gram 1 --c-n-gram 2-3
# keystrokes ext + embeds
python src/ml/classifier.py features/profiling/villani/linguistic+text.csv  --meta features/profiling/villani/label_map_users2gender.txt --embeds
#python src/ml/classifier.py features/authorship/villani/text.csv --c-n-gram 2-3 --n-gram 1
