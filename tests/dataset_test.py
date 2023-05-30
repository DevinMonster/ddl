from datasets import VOCIncrementSegmentation

if __name__ == '__main__':
    voc_inc = VOCIncrementSegmentation("../data/voc/")
    for x, y in voc_inc:
        x.show()
        y.show()
        break