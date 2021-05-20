echo "Choose model to train (e.g. SomDST)"
# echo "[SomDST, ChanDST, TransformerDST, TRADE_TAPT, SUMBT_TAPT]"
echo "[SomDST, ChanDST, TransformerDST, TRADE_TAPT]"
echo "=> "
read model_name

if [ $model_name = "SomDST" ]; then
    python ./SomDST/train.py
elif [ $model_name = "ChanDST" ]; then
    python ./ChanDST/main_chan.py
elif [ $model_name = "TransformerDST" ]; then
    python ./TransformerDST/TransformerDSTtrain.py
elif [ $model_name = "TRADE_TAPT" ]; then
    python ./TRADE_TAPT/train.py
# elif [ $model_name == "SomDST" ]; then
#     python ./SomDST/train.py
else
    echo "Invalid model name!"
fi

