import json


def main():

    train_data = json.load(open('./train.json'))
    test_data = json.load(open('./test.json'))
    val_data = json.load(open('./val.json'))

    out_data = train_data + test_data + val_data

    # for entry in out_data:
    #     print(entry)
    #     raise SystemExit()

    print(out_data[16])


if __name__ == "__main__":
    main()