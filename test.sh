# Question 11 -- Dropout after Batch Normalization
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q13UltimateNet --name q13_dropout_adam  --data_dir ../data/q13 --optimizer adam --epochs 20 --save_ep 

python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q13UltimateNet --name q13_dropout_sgd  --data_dir ../data/q13 --optimizer sgd --lr 0.1 --epochs 20 --save_ep 
