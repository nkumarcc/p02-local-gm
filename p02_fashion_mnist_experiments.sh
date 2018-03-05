#set -e
#set -u
#set -x

# Question 1 Default settings
 python p02_fashion_mnist.py --dataset mnist --epochs 10 --name q1_default
 python p02_fashion_mnist.py --dataset fashion_mnist --epochs 10 --name q1_default  --data_dir ../data/q1

## Question 2 -- Train for several epochs
python p02_fashion_mnist.py --dataset mnist --epochs 20 --name q2_mnist_20 --data_dir ../data/q2
python p02_fashion_mnist.py --dataset fashion_mnist --epochs 20 --name q2_fmnist_20 --data_dir ../data/q2


# Question 3 -- Change SGD Learning Rate
python p02_fashion_mnist.py --dataset fashion_mnist --lr 0.1 --name q3_lr_1 --data_dir ../data/q3
python p02_fashion_mnist.py --dataset fashion_mnist --lr 0.01 --name q3_lr_01 --data_dir ../data/q3
python p02_fashion_mnist.py --dataset fashion_mnist --lr 0.001 --name q3_lr_001 --data_dir ../data/q3

# Question 4 -- Change optimizer
python p02_fashion_mnist.py --dataset fashion_mnist --optimizer sgd --name q4_sgd --data_dir ../data/q4
python p02_fashion_mnist.py --dataset fashion_mnist --optimizer adam --name q4_adam --data_dir ../data/q4
python p02_fashion_mnist.py --dataset fashion_mnist --optimizer rmsprop --name q4_rmsprop --data_dir ../data/q4

# Question 5 -- Dropout
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 0  1
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 0.25 --name q5_dp_25 --data_dir ../data/q5
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 0.5 --name q5_dp_50 --data_dir ../data/q5
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 0.9 --name q5_dp_90 --data_dir ../data/q5
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 1 --name q5_dp_100 --data_dir ../data/q5


# Question 6 -- Batch Size
python p02_fashion_mnist.py --dataset fashion_mnist --batch-size 32 --name q6_32 --data_dir ../data/q6
python p02_fashion_mnist.py --dataset fashion_mnist --batch-size 256 --name q6_256 --data_dir ../data/q6
python p02_fashion_mnist.py --dataset fashion_mnist --batch-size 2048 --name q6_2048 --data_dir ../data/q6


# Question 7 -- Output Channels
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q7HalfChannelsNet --name q7_half --data_dir ../data/q7
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q7DefaultChannelsNet --name q7_def --data_dir ../data/q7
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q7DoubleChannelsNet --name q7_double --data_dir ../data/q7

# Question 8 -- Batch Normalization
python3 p02_fashion_mnist.py --dataset fashion_mnist --model P2Q8BatchNormNet --name q8_bn --data_dir ../data/q8

# Question 9 -- Dropout after Batch Normalization
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q9DropoutNet --name q9_bn_dp --data_dir ../data/q9

# Question 10 -- Dropout after Batch Normalization
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q10DropoutBatchnormNet --name q10_dp_bn --data_dir ../data/q10

# Question 11 -- Extra Convolution
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q11ExtraConvNet --name q11_extra --data_dir ../data/q11



# ...and so on, hopefully you have the idea now.
# TODO You should fill this file out the rest of the way!
