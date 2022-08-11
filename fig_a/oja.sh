LOOP=10
python3 main.py -algo oja -space exp1a_oja -gamma 0.0e-5 -loop $LOOP
python3 main.py -algo oja -space exp1a_oja -gamma 1.0e-5 -loop $LOOP
python3 main.py -algo oja -space exp1a_oja -gamma 3.0e-5 -loop $LOOP
python3 main.py -algo oja -space exp1a_oja -gamma 5.0e-5 -loop $LOOP
