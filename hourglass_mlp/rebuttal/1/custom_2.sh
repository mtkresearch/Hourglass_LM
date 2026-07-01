
PYTHONUNBUFFERED=1 ./rebuttal/1/mnist_generative_classification_aug2/run_L1_ep200.sh 2>&1 | tee ./rebuttal/1/mnist_generative_classification_aug2/run_L1_ep200.txt &

PYTHONUNBUFFERED=1 ./rebuttal/1/mnist_generative_classification_aug2/run_L2_ep200.sh 2>&1 | tee ./rebuttal/1/mnist_generative_classification_aug2/run_L2_ep200.txt &

PYTHONUNBUFFERED=1 ./rebuttal/1/mnist_generative_classification_aug2/run_L4_ep200.sh 2>&1 | tee ./rebuttal/1/mnist_generative_classification_aug4/run_L4_ep200.txt &

PYTHONUNBUFFERED=1 ./rebuttal/1/mnist_generative_classification_aug2/run_L6_ep200.sh 2>&1 | tee ./rebuttal/1/mnist_generative_classification_aug2/run_L6_ep200.txt &

PYTHONUNBUFFERED=1 ./rebuttal/1/mnist_generative_classification_aug2/run_L10_ep200.sh 2>&1 | tee ./rebuttal/1/mnist_generative_classification_aug2/run_L10_ep200.txt &