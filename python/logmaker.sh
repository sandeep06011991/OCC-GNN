mkdir -p logs
printf -v VAR 'logs/%(%Y-%m-%d-%H-%M-%S)T.log' -1
python3 $1 > $VAR