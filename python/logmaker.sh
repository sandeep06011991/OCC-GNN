printf -v DIRECTORY "logs/%(%Y-%m-%d)T" -1
printf -v FILE "$DIRECTORY/%(%H-%M-%S)T $1.log" -1
mkdir -p $DIRECTORY
python3 "$@" > "$FILE"