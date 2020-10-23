from network import *
from utils import *
from tqdm import tqdm


if __name__ == '__main__':

    params, cost = train()

    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    X, y = extractMNIST('./mnist/test')

    # Normalize the data
    X -= int(np.mean(X)) # subtract mean
    X /= int(np.std(X)) # divide by standard deviation
    


    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]

        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])]+=1

        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr / (i + 1)) * 100))

    print("Overall Accuracy: %.2f" % (float(corr / len(X)*100)))

    print(digit_correct)
