python3 scheduler.py -algo noisy -sigma 0.15 -delta 1.00 -loop 30 -label noisy_normal -earlystop
python3 scheduler.py -algo noisy -sigma 0.15 -delta 0.50 -loop 30 -label noisy_sdelta -earlystop
python3 scheduler.py -algo noisy -sigma 0.05 -delta 1.00 -loop 30 -label noisy_ssigma -earlystop

python3 scheduler.py -algo noisy -sigma 0.15 -delta 1.00 -loop 3 -label noisy_normal_series
python3 scheduler.py -algo noisy -sigma 0.15 -delta 0.50 -loop 3 -label noisy_sdelta_series
python3 scheduler.py -algo noisy -sigma 0.05 -delta 1.00 -loop 3 -label noisy_ssigma_series