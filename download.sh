
DATA_PATH="./data"

# Facebook babi task data v1.2
BABI_FNAME="tasks_1-20_v1-2"
BABI_URL="http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"

if [ ! -d $DATA_PATH ]
then
    echo " [*] make data directory"
    mkdir -p $DATA_PATH    
fi

cd $DATA_PATH

echo " [*] Download babi task dataset..."
if [ -d $BABI_FNAME ]
then
    echo " [*] babi already exists"
else
    { curl -O $BABI_URL; }
    tar -xvf "$BABI_FNAME.tar.gz"
fi
