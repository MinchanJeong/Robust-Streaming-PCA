LOOP=5
python3 main.py -algo noisy -space exp1a_noisy -gamma 0.0e-5 -loop $LOOP
python3 main.py -algo noisy -space exp1a_noisy -gamma 1.0e-5 -loop $LOOP
python3 main.py -algo noisy -space exp1a_noisy -gamma 5.0e-5 -loop $LOOP
python3 main.py -algo noisy -space exp1a_noisy -gamma 1.0e-4 -loop $LOOP
python3 main.py -algo noisy -space exp1a_noisy -gamma 5.0e-4 -loop $LOOP
python3 main.py -algo noisy -space exp1a_noisy -gamma 1.0e-3 -loop $LOOP