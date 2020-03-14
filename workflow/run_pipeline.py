from workflow import parse, preprocess, train, predict


def main():
    csv_file = parse.run()
    dataloader, num_classes = preprocess.run(csv_file)
    model_file = train.run(dataloader, num_classes)
    print('model trained under: {}'.format(model_file))
    predict.run(num_classes, model_file)


if __name__ == "__main__":
    main()
