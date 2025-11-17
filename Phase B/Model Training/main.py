import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

def main():
    print(keras.__version__)


if __name__ == "__main__":
    main()
