# mkdir -p logs
# printf -v VAR "logs/%(%Y-%m-%d %H-%M-%S)T $1.log" -1
# python3 $1 > "$VAR"

# mkdir -p logs
printf -v DIRECTORY "logs/%(%Y-%m-%d)T" -1
printf -v FILE "$DIRECTORY/%(%H-%M-%S)T $1.log" -1
mkdir -p $DIRECTORY
python3 "$@" > "$FILE"