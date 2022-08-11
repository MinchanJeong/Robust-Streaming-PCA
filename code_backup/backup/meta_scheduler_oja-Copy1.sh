python3 scheduler.py -algo oja -sigma 0.15 -delta 1.00 -loop 30 -label oja_normal -earlystop
python3 scheduler.py -algo oja -sigma 0.15 -delta 0.50 -loop 30 -label oja_sdelta -earlystop
python3 scheduler.py -algo oja -sigma 0.05 -delta 1.00 -loop 30 -label oja_ssigma -earlystop

python3 scheduler.py -algo oja -sigma 0.15 -delta 1.00 -loop 3 -label oja_normal_series
python3 scheduler.py -algo oja -sigma 0.15 -delta 0.50 -loop 3 -label oja_sdelta_series
python3 scheduler.py -algo oja -sigma 0.05 -delta 1.00 -loop 3 -label oja_ssigma_series