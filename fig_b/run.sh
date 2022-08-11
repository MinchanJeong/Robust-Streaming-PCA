LOOP=3
python3 1_scheduler.py -algo oja -space exp1b_finetunebase_oja -loop $LOOP
python3 1_scheduler.py -algo noisy -space exp1b_finetunebase_noisy -loop $LOOP
python3 2_generate_range.py
python3 3_scheduler_finetune.py -algo oja  -loop 30
python3 3_scheduler_finetune.py -algo noisy -loop 30
python3 4_plot_finetune.py