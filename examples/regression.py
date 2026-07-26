from s4_nnx import S4Regressor


def main():
    x_train = [0.0, 1.0, 2.0]
    y_train = [1.0, 3.0, 5.0]
    x_test = [3.0, 4.0]

    model = S4Regressor().fit(x_train, y_train)
    preds = model.predict(x_test)
    print(preds)


if __name__ == "__main__":
    main()
