
# Text
python src/ml/classifier.py --n-gram 1-2 --c-n-gram 2-3 features/profiling/villani/text.csv  --meta features/profiling/villani/label_map_users2age.txt
# keystrokes basic
python src/ml/classifier.py features/profiling/villani/letters.csv  --meta features/profiling/villani/label_map_users2age.txt
# keystrokes ext
python src/ml/classifier.py features/profiling/villani/linguistic.csv  --meta features/profiling/villani/label_map_users2age.txt
# keystrokes ext + text
python src/ml/classifier.py features/profiling/villani/linguistic+text.csv  --meta features/profiling/villani/label_map_users2age.txt --n-gram 1-2 --c-n-gram 2-3
# keystrokes ext + embeds
python src/ml/classifier.py features/profiling/villani/linguistic+text.csv  --meta features/profiling/villani/label_map_users2age.txt --embeds

