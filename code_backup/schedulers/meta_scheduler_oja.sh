LOOP=30
python3 scheduler.py -algo oja -sigma 0.15 -delta 1.00 -loop $LOOP -label oja_normal
python3 scheduler.py -algo oja -sigma 0.15 -delta 0.50 -loop $LOOP -label oja_sdelta
python3 scheduler.py -algo oja -sigma 0.05 -delta 1.00 -loop $LOOP -label oja_ssigma