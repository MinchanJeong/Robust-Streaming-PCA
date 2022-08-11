LOOP=30
python3 scheduler.py -algo noisy -sigma 0.15 -delta 1.00 -loop $LOOP -label noisy_normal
python3 scheduler.py -algo noisy -sigma 0.15 -delta 0.50 -loop $LOOP -label noisy_sdelta
python3 scheduler.py -algo noisy -sigma 0.05 -delta 1.00 -loop $LOOP -label noisy_ssigma